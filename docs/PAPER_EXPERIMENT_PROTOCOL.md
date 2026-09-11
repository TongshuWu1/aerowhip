# Clean aerial-whip experiment: protocol release candidate 1

11 September 2026. **Selected design; not yet released for clean collection.**
This protocol settles the intended baseline and reporting rules. Outstanding
hardware measurements and implementation checks are listed explicitly below.
The companion [audit](PAPER_PIPELINE_AUDIT.md) gives the theory, source inspection,
literature and development evidence. No fit, planner or flight is launched by
this document. No historical data role is reassigned.

## A. Fixed research scope

Study one repaired 145 g UAV with its 17 g cable assembly, the documented onboard
controller, and open-loop desired PVA execution. “Open loop” refers to the cable
task: onboard UAV tracking feedback remains active. The model is reusable across
commands and target locations within the tested envelope.

Primary question: does the complete real-to-sim-to-real process improve actual
target approach/interception after two model updates? The supporting model
question is whether the updated model predicts unseen commanded motions more
accurately. Neither question assumes monotonic improvement on every take.

Use MPPI-inspired offline whole-maneuver planning. Keep PPO outside the core
study. Do not claim a novel adaptation algorithm, exact MPPI importance-sampling
guarantees, identified motor limits, conserved learned dynamics, or measured
impact power.

Current development data remain in a separate namespace. The fresh paper study
starts with new preliminary observations and reset learned weights. Preserve
development artifacts for reproducibility, not for inclusion in clean result
tables or final-test fitting.

## B. Method selected for the study

| Element | Selected rule |
|---|---|
| Model | Effective loaded-UAV pose response → rotated attachment → discrete rod/cable dynamics |
| Model class | Same aircraft and cable residual architectures in M0, M1 and M2; zero-output networks at fresh M0 initialization |
| Mass/geometry | Survey once, freeze and hash; currently 145 g / 17 g and 0.9525 m cable; record whether individual mass values are measured or proportionally assigned |
| Initialization | 0.4 s causal aircraft history, 1 s causal cable history, weighted endpoint velocity with 0.02 s time scale; no measured resets inside a prediction window |
| Preliminary fit windows | Aircraft 2 s, cable 1 s; whole-take role assignment precedes window extraction |
| Whip fit window | Full frozen planned whip segment up to its declared end or first external contact, whichever is earlier; minimum usable coverage/length declared before fitting |
| Identification | Existing staged regularized simulation-error minimization; full aircraft nominal/residual/attitude and cable physics/residual stages |
| Optimizers | Bounded nonlinear least squares for positive nominal parameters; Adam with full temporal gradients for residuals; numerical verification and plateau rules from staged v1 |
| Data replay | New whip / all prior training whip / preliminary masses 1 / 0.5 / 0.5, normalized over present families; equal takes within each family |
| Warm start | Parent nominal parameters and networks; fresh optimizer state on a changed dataset |
| Selection | Best numerically valid training checkpoint, saved before validation; no validation-based checkpoint search |
| Combined error | Evaluate complete command-to-tip prediction after selection; do not describe this as joint training |
| Planner | 512 random samples per update, 1.5 s complete search horizon, 10 control points, fixed proposal families and exact versioned score/ranking |
| Planner restarts | Three prespecified seeds, 657/658/659, for each model/target condition; select by the frozen simulator ranking only |
| Search stopping | Current plateau procedure; no imposed wall-time cutoff. Record actual updates/candidates/time, rather than claiming identical compute |
| Seed/reference fairness | Same immutable command-seed bank and wave-style reference for every model; reroll every seed under that model |
| Task ranking | Feasible tip contacts ahead of misses; fixed preferred-fold score within each class |
| Contact-speed preference | Successful-contact bonus `1600*v_forward²/(16+v_forward²)`; no 4 m/s hard minimum or earlier-hit bonus |
| Execution | Exact 30 Hz PVA package with separately checked recovery; prepare the current state after planning, verify readiness, then launch |
| Primary real metric | Minimum 3D tip-to-target-center distance during `[0,1.5] s` from command onset |
| Secondary task metrics | Entry into the fixed 5 cm sphere; directed speed at first entry when observed; near-approach speed on misses separately labeled |
| Main model metric | Complete command-to-tip RMS on common unseen recordings, identical histories, commands, masks and windows across models |

These choices preserve the selected development objective as an engineering
baseline; they do not claim it is optimal. Use its exact source, settings, seed
commands and shape-reference files rather than rebuilding a reward from the
table alone. Before release, the manifest must contain those immutable hashes.
The three-restart comparison policy is a prospective design decision, not a
claim that historical jobs already used it.

The fresh preliminary-only full-model driver is still required. Its model class,
losses, initial priors and source must match this contract. It may omit unavailable
whip families, not silently invoke the old scalar/bootstrap method. No extra
joint-fitting stage is part of this release candidate.

## C. Preparation, execution and raw observations

1. Record hardware ID, cable marker geometry/masses, body-to-attachment transform,
   battery voltage, firmware/controller configuration and external sender version.
   Freeze controller settings throughout the study. A change creates a new
   experimental condition, not an undocumented adaptation round.
2. Survey the tracking frame and target coordinates. Record uncertainty and any
   origin changes. Verify the tracked origin and sender's commanded origin agree.
   Validate packet timing and onset against the actual execution/logging chain.
3. Compute and freeze the plan and its original nominal-state forecast. Preserve
   the full PVA, model, settings, source hashes, seed, score and recovery checks.
4. Move to the standardized preparation pose and allow the cable to settle.
   Evaluate readiness from current native pose and causal cable history after
   planning completes. Record the accepted launch state and all rejected attempts.
5. Launch the unchanged PVA when the verified readiness condition passes.
   Do not translate/rotate the plan at launch unless that transform itself becomes
   a tested, explicitly versioned part of the protocol before collection.
6. Record the original full native take, host/controller command log, battery
   state and synchronized overview/target video. Retain takeoff/landing in raw
   storage; hand-trimmed intervals are reviewed selections with a manifest.
7. Freeze uploaded raw bytes before any processing. Check command identity,
   marker identity, contact/intervention notes, clocks and geometric validity.
   Keep rejected/partial takes and their reasons. Never delete inconvenient misses.

The readiness specification must include position/orientation error, drone and
marker motion, marker availability, a dwell duration, state age at dispatch and
a timeout. These numeric tolerances are **not yet calibrated** and must not be
invented as verified aircraft capabilities. Use a short bounded development
check to set them, including the trade-off between repeatability and how often
a launch is possible. Freeze them before the clean collection. The observed
development range of 0.73–8.83 cm drone start error is not itself an acceptance
limit.

A virtual target avoids an unmodeled contact event during identification. If the
paper claims a physical strike, add independent contact evidence and explicit
target geometry. Exclude the post-contact interval from this free-cable model's
fit and prediction-error score; a physical contact changes the dynamics. Keep
the trial's task outcome, including a failed or interrupted execution.

## D. Data roles and the update loop

### Fresh preliminary collection

Collect several distinct excitation motions covering horizontal/vertical response
and direction changes in the intended command range. Assign whole takes to
training and holdout before looking at fit outcomes. At least one different
motion pattern should be held out; hundreds of overlapping windows from one
recording are not hundreds of independent experiments. The precise trajectories,
durations and repetitions belong in the release manifest after hardware review.

Fit fresh M0 once with reset networks and declared priors. Freeze all training
selection before evaluating preliminary holdout. Keep that holdout out of later
replay. Do not manually replace cable parameters between M0 and the first plan
without recording a method change and restarting the clean study.

### Two planned update opportunities

Use the adaptation target A and five recorded attempts per generation that
provides an update:

| Batch | Frozen plan/model used | Training roles | Operational validation roles | Result |
|---|---|---|---|---|
| Round 0 | M0 at target A | 001/002/004 | 003/005 | Fit candidate M1 from training only |
| Round 1 | Accepted M1 at target A | 001/002/004 | 003/005 | Fit candidate M2 from new and prior training only |

These prospective IDs apply only to the new study. Existing M2's three-take
001/002 training, 003 validation split remains unchanged.

For each batch, first produce the original-forecast comparison and real task
outcome. Prepare training observations using the frozen source-based rules.
Fit once from the recorded parent. Save the training-selected candidate and
numerical verification before reading operational validation metrics.

Operational validation is allowed to decide whether a candidate is used for the
next collection. It is therefore **not the untouched final test**. The selected
release-candidate adoption rule is: numerical/export checks must pass; on the
new operational-validation takes, mean complete tip RMS and mean drone RMS must
each be no more than 1.10 times the parent's error under the same inputs; and
preliminary-holdout mean drone and conditional tip RMS must each be no more than
1.10 times the parent's. Use a denominator floor equal to the independently
measured tracking uncertainty, recorded before release. Report every individual
regression even if the aggregate passes.

The 10% retention margin is a **proposed engineering tolerance**, not a statistical
significance test, measured sensor limit, or a rule applied to historical M1/M2.
Validate its operational meaning on development data once, then freeze it. Do
not retune it to make a clean candidate pass. The candidate must also produce a
plan within the reviewed execution envelope before a new flight is attempted.

If a candidate fails adoption, retain both parent and candidate, record the failed
update opportunity, and stop this adaptive chain under this protocol. Do not
quietly change bounds, remove a residual, borrow validation samples, restart
until a favorable fit appears, or rename the parent as an improved generation.
A subsequent revised method starts a new named study. This stop rule is needed
to make failed adaptation a visible result rather than a hidden search.

## E. Final real task comparison

Recommended main budget: **three frozen models × three target conditions × ten
repetitions = 90 final flights**, after the two five-take update batches. This
is a proposed budget, not a formal power calculation or a guarantee of sufficient
precision. If the available budget only supports five repetitions per cell,
45 flights form a useful pilot; qualify the uncertainty rather than promising
equivalent strength. Do not extend collection based on whether significance
has been reached.

Proposed target centers in the current world frame are:

| Target | Coordinates, m | Role |
|---|---|---|
| A | `[1.25, 0, 1.00]` | Adaptation target and final in-distribution condition |
| B | `[1.10, 0, 1.00]` | Untrained target-location condition |
| C | `[1.25, 0, 0.85]` | Untrained lower target-location condition |

These new B/C coordinates are design choices requiring workspace and numerical
feasibility review before release; they were not flown by this audit. Keep the
same 0.05 m radius and initial origin `[0,0,1.255]` m. If review requires other
coordinates, change the release candidate before any clean data, record the
reason, and freeze the final values. “Untrained target” means no fitting data from
that target; it does not establish new-hardware or arbitrary-task generalization.

For each of the nine model/target cells, generate the three prespecified planner
restarts with identical seed/reference provenance and select by simulator ranking
only. Freeze one selected plan per cell before final flights. Do not select a
command after seeing which one works on the real target.

Interleave the M0/M1/M2 plans in ten blocks per target, with randomized condition
order and recorded battery/session state. Balance order across batteries/sessions
so “M0 first on a full battery, M2 last on a depleted battery” is not the design.
Use a fixed order-generation seed in the final manifest. Every condition must
appear in multiple sessions if session variability is part of the claim.

Every final recording stays out of training, stopping, candidate adoption,
reward selection and launch-threshold tuning. The final evaluation is run after
the model/plan bank is locked. Record every scheduled attempt, readiness rejection,
execution abort and tracking failure. If a model has no acceptable planned
trajectory, report planning failure in that condition rather than selecting a
different method or discarding the cell.

A single M0→M1→M2 chain tests one learned system. Ten repeated flights of a frozen
plan do not create ten independently fitted models. Replicate complete chains
or hardware only if the paper claims that broader generality; it is optional
for the narrower system paper and must be described honestly.

## F. Endpoint definitions and analysis

### Real task outcome

For target center `g`, radius `r=0.05 m`, and tip `p_tip(t)`, define

```text
d_min = min over observed contiguous segments in [0,1.5] s of ||p_tip(t)-g||
entry = an observed contiguous segment crosses or enters the fixed sphere
```

Use native pose and the fixed segment/interpolation rule. Do not bridge gaps
longer than 1.5 native sample intervals. Report sampling/segment coverage and
tracking uncertainty. A positive observed crossing is evidence of geometric
entry. Without a crossing, classify **miss** only when the encounter interval
has sufficient valid coverage under the frozen rule; otherwise classify
**unknown**. Report `hits / observed attempts`, unknown count, and a conservative
`hits / all attempted executions` separately. Missing data do not establish a miss.

The shared 1.5 s task window can include the beginning of recovery for a shorter
whip. That is deliberate and identical across conditions. Report which segment
contains an encounter. Evaluate the complete recovery separately for execution
quality; do not extend the task window until a favorable later approach appears.

Directed entry velocity is a secondary metric evaluated only at first entry;
use the same target-forward direction and fixed local derivative estimator for
all real takes. Report total speed separately. On misses, call it near-approach
velocity, not impact. A force or impulse claim requires additional contact
measurement. Wave/shape diagnostics do not gate the outcome.

### Prediction accuracy

Keep two named evaluations:

1. **Original prospective forecast:** compare the frozen forecast made before
   execution with the actual take. Include its launch-state assumption. This
   measures the operational pipeline and must remain available unchanged.
2. **Matched initialized prediction:** roll every frozen model on the same unseen
   recording, command schedule, causal state and masks. Report drone position,
   conditional cable tip under measured attachment, and complete command-driven
   tip. Primary model endpoint is complete tip RMS, not the conditional result.

Score the common fixed interval for final flights and report coverage/contact
censoring. Do not make the interval stop at each model's predicted hit, which
would compare different durations. Retain the historic development intervals
in their original reports; the new common interval belongs to this protocol.

The primary comparisons are final M2 versus M0 real `d_min`, and paired M2 versus
M0 complete prediction RMS on common final observations. M1 provides the
intermediate point. Report individual values, medians/means, effect sizes and
uncertainty respecting the session blocks. Do not use individual frames as
independent trials. Report binary outcomes with denominator and unknown count;
do not hide poor target conditions inside an aggregate.

Any confidence interval or statistical model must be named before final analysis.
A practical primary analysis is paired within-block distance differences with
session-aware uncertainty, and paired per-take model-error differences clustered
by recording session. With too few independent sessions for reliable resampling,
show block/session results directly and qualify precision. A development-informed
precision/budget check is allowed before release, not after inspecting the final
effect. Hypothesis tests are secondary to the magnitude and consistency of the
measured effect.

## G. Necessary versus optional ablations

Necessary for the chosen system claim: unchanged-planner real M0/M1/M2 comparison,
matched-model complete prediction, component diagnostics, initialization/coverage
reporting, and numerical replay/step-size checks. These directly address whether
the real-to-sim-to-real loop helps.

If claiming that neural residuals are essential, add a separately retrained
nominal-only baseline using the same data and fixed selection rules before final
testing. Turning off a trained residual is only a removal diagnostic. If claiming
the wave-style prior or strong speed bonus improves the task, evaluate its
prespecified ablation; otherwise describe it as a fixed design choice. Warm start
versus cold all-data refitting, PPO versus MPPI, multiple-shooting variants and
joint fitting are not mandatory novelty experiments for this system claim.

## H. Release checklist and outstanding fields

The machine-readable design is
[`paper_protocol_rc1.json`](../runs/audits/paper-pipeline-audit-20260911/paper_protocol_rc1.json).
It is documentation, not an executable experiment registration.

| Release condition | Status at this audit | Completion evidence required |
|---|---|---|
| Consistent fresh M0 and full update path | Open | Development-only full-scope M0 dry run; same model class/losses; no legacy scalar defaults |
| Actual sender/controller/origin/timing contract | Open | Version/parameter export, execution test and measured clock/geometry uncertainty |
| Live preparation/readiness contract | Open | Tested current-state gate; numeric thresholds, dwell, age and timeout fixed |
| Representative numerical sensitivity | Partial | Selected M2 passed outcome stability; add preliminary and another fast motion with a declared tolerance |
| Target reconstruction and missing-data policy | Partial | Current gap-aware code tested; finish uncertainty and encounter-observability specification |
| Candidate retention margin and denominator floor | Proposed | Confirm the stated engineering rule and measurement floor before collection |
| Targets and operating envelope | A established in development; B/C proposed | Survey/workspace review and frozen bounds; no late target movement |
| Fresh preliminary motion schedule and roles | Open | Named take IDs, motion definitions/durations and holdout plan |
| Final order/budget/analysis specification | Proposed | Fixed 90-flight budget, block/session allocation, order seed and uncertainty procedure |
| Complete manifest | Open | Source, model architecture, seed/reference, firmware, hardware and method hashes |

The release operation sets `ready_for_clean_collection: true` only after those
facts exist. This is an evidence requirement, not a reason to ask the user to
approve already authorized read-only work. The next implementation task should
close these release gaps on development data; it should not start a new M3 fit
or silently use the clean final recordings for development.
