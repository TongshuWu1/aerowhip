# Fixed-reference command correction

Decision recorded 13 September 2026, before prospective M1 executions.
The user selected: prioritize the original predicted cable-tip motion, while
keeping the quadrotor near its original predicted path. This replaces replanning
a different whip after each model update for the next physical comparison.
The completed M0 flights and their original analysis remain unchanged.

## What remains fixed

The reference is the **original M0 prediction**, saved in
`runs/rehearsals_pva/20260913-012740-484590-M0-slower-brake-1s/rehearsal.npz`.
Use its predicted physical tip and tracked-origin positions, with their original
timestamps. The PVA commands in the CSV are inputs to the onboard controller;
they are not the predicted physical quadrotor trajectory.

- Tracked-origin launch: `[0, 0, 1.4]` m.
- Physical target: `[1.25, 0, 1.25]` m.
- Original planned strike time: `1.1172482457473654` s.
- Fixed maneuver/handover time: `34/30 = 1.1333333333333333` s.
- Original command spline: quintic, 12 control points, original 1.5 s knot
  domain. Only the prefix through the fixed handover was executed.
- Same 150 Hz model grid and 30 Hz PVA command interpretation.
- Original measured M0 results, model assets, commands and forecasts.

For correction, score the fixed interval from launch through handover. Do not
stop early when a simulated tip enters the target sphere, select a different
strike time, shift trajectories in time, or substitute a newly predicted
trajectory for the reference. Preserve the full original recovery forecast for
diagnostics, but do not force the tip to track the original recovery motion.

## Command correction

After M1 is frozen, simulate candidate commands with the complete M1 quadrotor
and cable model. Adjust the free position-spline controls, starting from the
original M0 controls. Regenerate P/V/A together from the spline derivatives.
The original settled initial state remains the planning initial condition;
do not initialize prospective commands with a held-out flight's cable state.

The intended objective contains three terms:

1. Time-matched mean squared error to the original predicted tip position.
2. A smaller time-matched mean squared quadrotor-position error.
3. A small penalty on command-position changes relative to the original CSV
   prefix, to discourage unnecessary compensation.

All three are squared distances in square meters. A quadrotor tracking penalty
is a soft preference, not a claim that every deviation lies below a bound.
Record both average and maximum deviations. Tracking the tip at fixed times
also constrains its timing; no separate reward for carrying the cable forward
or searching for a new high-speed strike is needed in this correction stage.
The original planner reward remains part of how the initial whip was designed.

Use the existing sampling optimizer and B-spline representation. Freeze the
numerical weights, proposal settings and compute budget in a new correction job
before running it. These values are not yet selected by this decision record.
Do not tune them against operational-validation takes or future final flights.
The unchanged command under M1 is the zero-correction simulation baseline.
If no feasible correction improves the declared objective, retain that result;
do not label a failed search as an improved plan.

Keep the saved command, jerk, workspace and model-validity limits. Append the
existing slower brake, return and hold from the corrected handover P/V/A using
the same recovery settings (minimum brake 1 s). Check the complete command and
coupled prediction through recovery before producing a candidate CSV. Existing
checks do not certify cable-to-propeller clearance. Offline output is not a
flight sender or physical flight authorization.

## Prospective comparison

The sequence is: original M0 plan -> physical M0 executions -> fit M1 -> correct
commands toward the **same** reference -> physical corrected executions. If a
later update is collected, keep the same reference again. Improvement in real
motion is measured, not guaranteed by the model-based optimization.

Report separately:

- Physical tip and quadrotor tracking RMSE against the frozen original
  reference over `[0, 34/30]` s, using measured-clock alignment and valid samples.
- Physical tip-to-target distance at the original planned strike time.
- Closest physical tip-to-target distance over the unchanged `[0, 1.5]` s window.
- Common-initialization M0/M1 model prediction errors on the same held-out logs.

Show every take. Do not time-warp trajectories or normalize away spatial errors.
The original reference itself misses the target center by a finite distance;
better reference tracking does not mathematically guarantee lower target error.
This experiment tests coupled-model command compensation, not a newly optimized
task trajectory. It is related to model-based iterative learning control;
fixed-reference correction alone should not be claimed as a new control idea.

Relevant precedent: A. P. Schoellig, F. L. Mueller and R. D'Andrea,
"Optimization-based iterative learning for precise quadrocopter trajectory
tracking," Autonomous Robots 33, 103-127 (2012),
DOI: 10.1007/s10514-012-9283-2. No BibTeX entry has been synthesized.

## Implementation status

`tools/freeze_paper_motion_reference.py` preserves the selected reference and
its source hashes without modifying M0, fitting data or active planner settings.
`tools/correct_paper_reference.py` now performs the correction with the finalized
M1. The first correction job is
`runs/reference_tracking/M1-20260913-030505-743382`. Its settings and source hashes
were frozen before search: tip/quadrotor/command weights `1, 0.1, 0.01`, 512 random
candidates per update, 4 proposal groups, 38 updates, seed 657, and the retained
5/15/40 mm position-control noise scales with spline-derived correlation.
The objective uses every 150 Hz sample through the fixed handover; no target
event truncates it. The previous M0 controls initialize all groups. Numerical
weights are engineering choices, not physical success thresholds.

M1 training was stopped by the user after saved update 375. Best update 370
passed the unchanged numerical check and was frozen before held-out evaluation.
No optimizer updates occurred during finalization. M1 is registered as a
candidate; it has not been flown. The correction is launched by the above CLI;
existing MPPI search buttons still invoke the original task planner. Completed
corrections appear in Rehearsal and report fixed-reference errors.
