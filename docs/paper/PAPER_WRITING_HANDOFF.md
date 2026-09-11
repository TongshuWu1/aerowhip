# AeroWhip paper-writing handoff

**Title:** AeroWhip: Aerial Cable Whipping through Iterative Model Refinement.
**Repository:** `aerowhip`.
**Current study:** [retained-M0, one-target, 20-flight protocol](PAPER_EXPERIMENT_PROTOCOL.md).

## Contribution and method

The contribution is the complete aerial dynamic-whipping system: an effective
loaded-aircraft/cable predictor, offline planning and measured model refinement
between flights. Standard identification supports the system; the paper does
not need to claim a novel adaptation algorithm.

Describe the implemented chain in this order:

1. Bounded XYZ jerk integrates into desired PVA commands at 30 Hz.
2. Held packets drive the fitted loaded-UAV pose response and its residual.
3. The rotated tracked-origin-to-attachment offset drives the cable model.
4. MPPI-inspired whole-maneuver search selects a simulated trajectory. Its
   recovery and final hold are exported with the frozen command and forecast.
5. Reviewed measured flights update aircraft response/residual and cable
   physics/residual in stages; a new trajectory is planned in the updated model.

The current estimator is staged, regularized nonlinear system identification
by simulation-error minimization: bounded nonlinear least squares for nominal
response/physics and Adam with temporal gradients for neural corrections.
Read [the frozen method](../methods/FROZEN_SYSTEM_IDENTIFICATION.md) for exact loss scales,
weights, bounds, stopping and initialization. Its older fresh-M0 proposal is
superseded: retain the existing M0 and preliminary recordings.

Use “offline MPPI-inspired trajectory optimization.” The code is not exact
path-integral control or cable-feedback MPC. The empirical loaded-UAV model
does not add explicit cable-reaction feedback or identify motor limits.
Combined command-to-tip error is evaluated after staged fitting, not jointly
optimized. The flight program is external to this repository.

## What evidence exists now

Existing real-flight development evidence follows **M0 → M1-full →
M2-frozen-refit-v1**, with **5/5/3 takes**. The gain-only M1 is a different sibling.
The original full M2 and its frozen refit share a model signature; do not count
them as two updates.

Development reports contain original-forecast errors, common-flight model
comparisons, fitting diagnostics, numerical checks and selected simulated plans.
They show mixed per-take behavior and motivated subsequent choices. They are
not an untouched final test and should not fill the new study's results table.
Use their saved reports when discussing development; do not copy aggregate
numbers without their exact windows, masks, lineage and command conditions.

Useful evidence sources:

- [Development system comparison](../development/M0_M1_M2_SYSTEM_COMPARISON.md)
- [Full-model adaptation](../methods/FULL_MODEL_ADAPTATION.md)
- [M2 regression mechanisms](../development/M2_REGRESSION_ANALYSIS.md)
- [Whole-system theory/implementation audit](PAPER_PIPELINE_AUDIT.md)

PPO and two-target experiments are separate development work. They need not be
part of the core paper. PPO status comes from the actual job output, not dated
documentation. A PPO/MPPI comparison requires matched model, objective, priors,
training/search budget and execution conditions before supporting a fair claim.

## Minimal paper results

| Result | Data | Presentation |
|---|---|---|
| Physical target approach | Five final paired M0/M2 blocks | Every minimum 3D distance, paired difference and coverage |
| Prediction accuracy | All frozen M0/M1/M2 models on the same final recordings | Per-recording command-to-tip RMS, plus drone/conditional diagnostics where available |
| System operation | Exact saved command, original forecast and a reviewed measured take | One clearly labeled example and the plan–measure–refine workflow |
| Experiment cost | Recorded collection and computation logs | Executions, fitting/planning time, failures and stopping reasons |

Keep the main visual set small: a system/method figure, the paired distance
result, and the common-recording model-error result. An illustrative trajectory
must state how the example was selected; label a best case as a best case.
Missing samples and excluded intervals remain visible. Frames are not
independent experimental repetitions.

## Interpretation that must stay explicit

- Primary task outcome is continuous minimum 3D distance during 0–1.5 s. There
  is no paper requirement to be within 5 cm. Do not confuse geometric approach
  with measured physical contact, force, impulse or impact power.
- Preserve M0's cable-residual-disabled configuration. Full M1/M2 enable that
  residual, so the study evaluates the full refinement procedure including
  capacity change; it does not isolate the effect of extra data alone.
- Final M0 trials are new repeats of a command represented during adaptation.
  Call final M2 commands unseen only after checking that the sequence differs
  from both training-stage commands.
- Original nominal-start forecasts and common-history postflight predictions
  answer different questions. Keep their tables, labels and intervals distinct.
- One target and one model-update chain support a focused system demonstration,
  not broad task, hardware or independent-chain generalization. Report all
  individual differences; improvement is a result to measure, not an assumption.
- Command logs and estimated clock offsets do not prove onboard timing or
  independent synchronization. Report the actual sender, hardware, controller,
  frame/origin, launch preparation, tracking coverage and any intervention.
- Windows/RTX 4080 software checks do not establish Ubuntu/RTX 5080 validation.

The introduction, methods and experimental design can be written now. Leave
new quantitative results and outcome claims pending collection and analysis.
Verify literature claims against primary sources using the
[current literature review](ICRA_2027_RELATED_WORK_AND_FRAMING.md),
[reference notes](RELATED_PAPERS_DETAILED_REVIEW.md) and
[bibliography](related_papers.bib).
