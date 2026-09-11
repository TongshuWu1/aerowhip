# M0 flight → reviewed physical M1 update

The current project-level scope is [drone, cable and residual adaptation](../methods/ADAPTATION_MODEL_CONTRACT.md).
The scalar procedures below remain evidence of the implemented limited trials;
they must not be treated as a complete implementation of that broader workflow.

Current result: [first real adaptation review](M0_M1_FIRST_ADAPTATION.md). A reviewed
single horizontal drone response gain was fitted after diagnosis; cable/NN/delay
stayed fixed. M1 is an unselected candidate because held-out 005 errors increased.
No M1 flight or automatic follow-up is authorized by completion. The scalar-cable
workflow below remains available for separately justified cases; statements that
the response branch is unimplemented predate the new gain-only contract.

The [drone-command chain audit](../methods/DRONE_COMMAND_CHAIN_AUDIT.md) verifies the exact
selected CSV/forecast mapping and flags command-acceleration extrapolation.
References below to command "receipt" mean the reconstructed logger command
clock, not confirmed onboard reception. The actual sender/logger implementation
must establish timestamp semantics before physical delay claims.

For repeated rounds and the new UI, read [sim-real evaluation](../methods/SIM_REAL_EVALUATION.md).
New candidates carry generation indices and checksums for completed candidate
assets/diagnostics. New batch setup can use `--selection` for the next frozen
flight package. These additions preserve the review gates below; no real M1
is currently fitted. Diagnostic coverage now excludes initialization frames
outside the scored interval. Existing saved results are not rewritten.

Prepared 10 September 2026 for selected MPPI run `20260910-022818-648386`.
This is a preparation and software audit. No new real flight data exists yet,
and no real M1 has been fitted or selected.

## Integrated decision before the first whip flight

The software checks support the identity and reproducibility of the selected
MPPI package, not an unconditional full-strength flight clearance. Its peak
command acceleration is 14.31 m/s²; the preliminary drone-fitting commands
reached 8.18 m/s². The latter is a data-coverage limit, not a measured vehicle
limit. The actual sender/logger path, PVA/frame/timestamp semantics and same
controller configuration still need verification. Resolve these and establish
appropriate physical execution precautions for the actual experiment. We do not
need M1 or perfect tracking before an M0 experiment. Exceeding the training-data
maximum alone is not a reason to reject or clip a command; it identifies where
the forecast is extrapolating and new measurements are informative.

If a lower-demand execution check is needed for physical testing, prepare it
separately; it is not a new acceleration gate in the planner. Never slow or edit the frozen CSV in place:
retiming changes velocity and acceleration too and requires a new command,
model rollout, recovery check and forecast identity. No such replacement was
created by this audit. Controller/gains, masses, selected CSV and forecast remain
unchanged. The flight sender must implement the documented initial hold and
takeoff/landing outside the CSV as well as its complete recovery, not just its
1.13-second whip prefix.

## Combine drone response and cable adaptation without a joint blind fit

The [drone-response literature/math review](../methods/DRONE_RESPONSE_ADAPTATION.md) now
provides a tested `diagnose-drone` stage: recursive pose error, retrospective
native-pose derivative plots, and local gain/delay/attitude-lag sensitivity.
It reads adaptation takes only and does not publish a drone candidate. Read its
explicit implementation boundary before interpreting this as saturation fitting.

User clarification: learn the achievable command-to-motion response in adaptation;
do not introduce a hard cap on generated trajectory acceleration based on the
preliminary-data maximum. MPPI should propagate the learned drone response and
then the cable, scoring predicted executed motion. A large command that produces
little additional drone response must not produce fictitious cable energy in
the model. This changes our model-design goal, not the frozen M0 or current
planner settings. Existing operational and numerical checks are not removed.

The intended learned mapping is next drone state from current state, command
history and available measured conditions. For example, if a command requests
14 m/s² while repeat measurements under those conditions show only about 9 m/s²,
the model should predict the observed response there. It must not relabel 9 m/s²
as a universal hard ceiling from that observation alone. Response gain, lag and
smooth saturation are candidate explanations to distinguish using data; a
saturation model belongs inside the drone dynamics, not as command clipping.
The numerical example is illustrative, not a measured result from this system.

Record three distinct signals: actual logged commanded PVA, the original M0
predicted drone/cable motion, and native measured drone pose/cable markers.
Retain the initial hold, full recovery, clocks, command validity and exact flown
file. Record controller/settings identity and battery voltage when available.
Cached controller XYZ is derived from OptiTrack and is not an independent sensor.

| Comparison | What it answers |
| --- | --- |
| Measured drone minus commanded drone | Tracking response of the closed-loop vehicle. Nonzero error alone does not imply an inaccurate model. |
| Measured drone minus M0-predicted drone under those commands | Drone-model mismatch, after checking clock, frame and initial-state errors. |
| Measured cable minus prediction supplied with measured rotated attachment | Conditional cable-model mismatch, with drone prediction removed from the boundary input. |
| Measured cable minus fully command-driven prediction | Whether the combined simulator predicts the actual execution. |

Use adaptation takes 001/002 to choose the update scope. First review timing,
tracking origin/rotated attachment, masks and causal initialization (1 s cable,
0.4 s drone). A failure here is a data/state-estimation issue before it is a
parameter-fitting issue. Preserve raw coordinates; no retrospective height
normalization enters this workflow.

If drone prediction dominates the mismatch, the next justified extension is a
small closed-loop drone-response update, chosen from evidence such as an
effective response gain or delay. Do not fit gains, delay, saturation and a
neural residual simultaneously from one repeated whip. Fit recursive pose
trajectories (and velocity where reliable), using recorded commands. Acceleration
diagnostics need reviewed smoothing on contiguous valid native-pose intervals,
with gap/estimator-edge neighborhoods excluded; naive second differences of
cached XYZ are unsuitable. A single tracking discrepancy cannot identify a
maximum acceleration: direction, loading, voltage, feedback and timing can all
affect it. Saturation needs repeated varied-demand evidence before adding a
fitted limit. Describe any resulting bound as an effective response envelope
over the observed conditions, not a certified actuator limit.

**The drone-update branch is a design, not an implemented M1 fitter.** The
current prepared protocol, fitter and same-flight model comparison allow scalar
cable damping changes only. A justified drone update requires a new reviewed,
versioned preparation/fitting/comparison contract; do not bypass these checks.
Keep geometry and neural weights frozen initially. For a future changed-drone
comparison, measured initial pose/velocity/orientation must use the same past
observations for every model; estimate any model-dependent nuisance bias using
past-only data and report it. It is not the firmware integral state.

If measured-attachment cable error supports a cable update, the existing GPU
scalar-damping workflow below is available. If both components need changes,
identify the drone response separately, then condition cable fitting on measured
attachment; check the resulting combined command-driven prediction. Cable damping
must not compensate for drone tracking errors. Keep M0 if the evidence supports
neither update; completing an optimizer is not a reason to promote a model.

Freeze the candidate before looking at parent/candidate validation diagnostics
on take 003. The original-forecast comparison remains separate M0 evidence;
if its validation results influence development choices, disclose that the take
is no longer an untouched test. Do not repeatedly tune on 003. Registration is
not promotion. A new frozen M1 plan and a later measured flight are the next
prospective sim→real check; repeated M0 commands alone test repeatability.

This staged design follows the existing primary-source review below and in
[the evaluation guide](../methods/SIM_REAL_EVALUATION.md). SimOpt motivates alternating real
rollouts and simulator adaptation, not our particular parameterization or a
requirement to fit every component at every round.

## Tomorrow's data

The selected command and forecast remain frozen in
`runs/flight_packages/20260910-022818-648386` and the original rehearsal.
The new inbox is:

`data/flight_batches/M0_whip_20260910-022818-648386/flight_take`

Put native OptiTrack `whip_001.csv` and controller `experiment_whip_001.csv`
together there. Repeat with `002` and `003` if practical. Retain original full
recordings, initial hold and recovery; record any physical target contact,
intervention, hardware/controller change and the exact command file flown.
The inbox already contains a byte-identical copy of the selected full command.
Do not overwrite preliminary1 or replace the original predicted NPZ.

The protocol predeclares takes 001 and 002 for adaptation, 003 for validation.
This is a small useful split, not a requirement to collect five flights.
Do not discard a miss just because it missed. Exclude unusable recordings with
reasons; do not move a validation take into fitting after seeing its error.
Other filenames require explicit whole-take roles. With only one usable take,
M1 fitting is a development exercise and supplies no independent M1 validation.
A repeated execution of the same command tests repeatability, not new-motion
generalization. A later M1-planned flight is the prospective next check.

## What the audit found and corrected

| Finding | Action |
|---|---|
| The old combined fitter is hard-coded to five historical takes, a one-second old CSV, a historical model and retrospective hover-Z correction. | Keep it historical. The new entry point is `tools/adapt_whip.py`, tied to this selected M0. The UI blocks sending prospective batches into the historical neural fitter. |
| Adaptation Check forced normalized Z and refused raw data without a hover calibration. | Raw global XYZ is now the default. An existing correction is available only as an explicitly selected retrospective diagnostic. |
| Repeated identical command values could be treated as unique timing anchors; negative command age was accepted for onset detection. | Both are excluded from timing anchors. Too few unique matches or inconsistent receipt times block comparison. |
| Dropping invalid controller rows could silently extend the preceding hold in fitting inputs. | The new protocol explicitly ends command coverage at invalid rows. Historical prepared-data semantics remain unchanged. |
| Old fitting had 0.2-second cable initialization and phase times tied to the old maneuver. | Use the requested one second of past cable history and 0.4-second drone history. The scored end comes from this rehearsal and reviewed contact timing. |
| An automatic joint drone/cable/neural fit could make one component absorb errors from another. | First report the frozen forecast, then compare conditional cable and command-driven diagnostics. The prepared first update changes one cable damping scalar only, after a separate diagnostic review. |
| A physical target collision is outside the free-cable simulator. | Fit only the reviewed free-motion prefix. Never train this model on collision/contact or manual-intervention dynamics as though they were cable damping. |

Raw logs, source hashes, tracking identity, masks, command timestamps, selected
forecast and model assets are preserved. No masses, geometry, controller,
selected model, MPPI reward or flight command was changed by this work.

## Research basis and limits

Shen, Franchi and Gabellieri identify two cable coefficients from a controlled
release while measuring geometry and density separately. This supports using
few physical parameters and an explicit initial condition. Their flexible,
extensible cable model and fixed-drone release differ from our bending model
and aerial whip; their numerical coefficients do not transfer to our cable.
[Aerial Robots Carrying Flexible Cables, VI-B](https://arxiv.org/html/2403.17565v2).

Mamedov and colleagues distinguish training-time initial-state optimization
from test initialization using preceding observations. Their work also warns
that a learned state recognizer can overfit small datasets. We retain a fixed
causal initializer instead of adding an estimator network or adjusting initial
states using scored future motion. Our estimator is not their moving-horizon
optimization, and neither approach guarantees accurate initialization here.
[Learning deformable linear object dynamics from a single trajectory, 4.3](https://arxiv.org/html/2407.03476v1).

DEFORM uses recursive one-second training and evaluates five-second forecasts
without later ground-truth state input. It uses 350 seconds of data per object
and a different boundary-control setup. The relevant principle is testing
recursive prediction beyond the training conditions; its data scale does not
justify adding a new residual network to one or two whip recordings.
[Differentiable Discrete Elastic Rods, 5.1](https://arxiv.org/html/2406.05931v2).

SimOpt iterates real rollouts and simulation updates. It adapts a parameter
distribution alongside policy training, whereas our immediate experiment is a
small physical identification update. We borrow the iterative experimental
logic, not the full algorithm or an assumed guarantee of improvement.
[SimOpt author project](https://sites.google.com/view/simopt/home).

The three arXiv methods sections above were checked directly in this audit;
SimOpt was checked through its primary author project/abstract. These sources
motivate the design, not our numerical thresholds. The 1 s history, 20 ms
velocity weighting, 2 cm robust-loss scale, 80% observation coverage and damping
range below are explicit project choices. Their suitability must be reviewed
on the new data, not described as paper-established constants.

## The five explicit steps

1. **Establish timing and inspect the pair.** Use the same global OptiTrack frame,
   exact rigid-body identity and rotated tracked-origin→attachment offset.
   `time_alignment.json` records `controller time = OptiTrack time + offset_s`,
   the evidence and hashes of both files. Shared timestamps/events are preferred.
   If only cached controller XYZ is available, matching it to native OptiTrack
   estimates stream alignment, including unknown cache/transport latency; it is
   not independent synchronization or actuator-delay measurement. The existing
   measured-stream alignment helper can support a reviewed estimate. Never align
   the measured cable to a predicted target hit or fit an offset to improve the
   model score. Missing/ambiguous timing remains a blocker, not a guessed zero.

2. **Score the frozen M0 forecast first.** `compare` reads the original prediction;
   it does not simulate a replacement or normalize height. Save raw drone/tip
   errors, coverage, native target distances, clock evidence and hashes. A
   sampled distance is not proof of physical contact, and missing tip samples
   are not automatically misses. The report and original forecast stay separate
   from all post-fit diagnostics.

3. **Review and freeze the fitting inputs.** Complete `review.template.json` into
   a separate reviewed JSON. Confirm hardware/controller identity, timing,
   intervention status, whole-take roles and a conservative free-motion end.
   For a physical strike, choose an end before contact uncertainty begins.
   Preparation uses actual logged command receipts and validity coverage; it
   does not silently replace missing commands with ideal packets. Drone and
   cable gaps/jumps remain masked; no interpolation across missing samples or
   time gaps. Initial state uses past samples only. A missing clean one-second
   cable history or missing attachment trajectory blocks conditional fitting.
   Raw values remain saved alongside masks. Model assets, code and input hashes
   are frozen; later changes require a new prepared job.

4. **Diagnose before choosing the update.** `diagnose` compares M0 on adaptation
   takes only, against the reviewed interval using measured initialization. One replay supplies
   measured attachment motion to isolate cable prediction. The other supplies
   recorded commands to the drone model and then its predicted attachment.
   These reinitialized diagnostics are not the original prospective forecast.
   Inspect the initial projection correction and missingness too. If command-
   driven error dominates, investigate drone response, attachment orientation,
   timing and initial state instead of changing cable damping. A separate
   `fit_review.template.json` binds the decision to these exact diagnostics.

5. **Make and check one small candidate.** Only if justified, `fit` adjusts
   `cable.external_drag_s_inv` within [0, 2]/s, starting from M0's 0.4/s.
   Geometry, masses, EI/Cb, smoothing, drone parameters and residual weights stay
   fixed; the cable NN remains disabled. This is effective velocity damping,
   not a measured aerodynamic coefficient. Whole adaptation takes receive equal
   weight; the objective equally combines robust marker and tip position loss
   over the reviewed free-motion interval. There is no hit-reward fitting.

The nine-row search retains M0 and the incumbent while refining a bounded
one-dimensional interval. Candidates and takes are batched on CUDA using the
same float64 physics and captured kernels as the checked simulator. There is
no differentiation requirement for this scalar search. Practical plateau uses
0.5% meaningful improvement and three stale checks after at least three updates;
12 updates is a separate safety ceiling. A STOP file stops at an update boundary.
History, best scalar and search state are saved; no automatic resume occurs.
A boundary optimum, flat loss or negligible change is reported, not presented
as globally converged identification.

After freezing the scalar, an independent batch-one replay must agree with its
selection loss. `fit/selection_frozen.json` records the candidate checksum before
reinitialized parent validation is evaluated in `baseline_validation`; candidate
results are saved in `candidate_diagnostics`. Candidate conditional and command-driven
errors use the same initialization/masks as M0. Validation takes enter neither
the pre-fit model diagnosis nor the optimizer. Check
both comparisons before selecting a candidate; a conditional fit improvement
alone is insufficient. The candidate remains unselected and marked development.
No new MPPI, neural training or flight launches automatically. A fresh M1 plan,
frozen forecast and subsequent measured flight complete the next research step.

## Commands for the next Codex session

Run from the project root. The inbox is already prepared; do not rerun setup.
Timing and reviews must describe the actual new files; never fill them blindly.

```powershell
$batch = 'data/flight_batches/M0_whip_20260910-022818-648386'
$comparison = 'runs/data_review/M0-whip-first-comparison'
$job = 'runs/adaptation/M1-whip-first'

.venv/Scripts/python.exe tools/adapt_whip.py compare --batch $batch --output $comparison
# Review the plots in Adaptation Check and complete a separate review.json.
.venv/Scripts/python.exe tools/adapt_whip.py prepare --batch $batch --comparison $comparison --review "$comparison/review.json" --job $job
.venv/Scripts/python.exe tools/adapt_whip.py diagnose-drone --job $job --output "$job/drone_diagnostic"
.venv/Scripts/python.exe tools/adapt_whip.py diagnose --job $job
# Review baseline/metrics.json and complete a separate fit_review.json.
.venv/Scripts/python.exe tools/adapt_whip.py fit --job $job --review "$job/fit_review.json"
```

Use new output names if a previous comparison/job already exists. Failed or
partial preparation is preserved; it is not overwritten. A changed source or
clock alignment requires a new comparison and reviewed job.

## Verification completed tonight

Focused regressions cover raw/default and optional normalized views, exact CSV
and clock identity, ambiguous/repeated command timing, missing samples, past-only
history, review gates, fixed masks/equal take weighting, and blocking the legacy
neural workflow for this batch. A synthetic end-to-end RTX 4080/Windows float64
check constructs native-format paired logs from the frozen forecast, preserves
an injected marker gap, prepares a whole-take split, diagnoses and runs the
scalar fitter. A separate known-parameter check distinguishes 0.7/s from 0.4
and 1.0/s; a modified prepared file is rejected. The synthetic no-change fit
retains 0.4/s rather than inventing improvement. Original flight artifacts are
hash-checked afterward. These tests do not establish physical M1 performance.

Evidence: `runs/audits/M0-to-M1-readiness-20260910`.

Follow-up integrated audit: `runs/audits/integrated-adaptation-20260910`.
Thirty focused tests and a fresh Windows/RTX 4080 synthetic end-to-end check
passed after restricting pre-fit diagnosis to adaptation takes. The synthetic
check asserts validation evaluation follows the frozen candidate selection.
Selected and retired evidence hashes remain unchanged; no real M1 fit ran.
