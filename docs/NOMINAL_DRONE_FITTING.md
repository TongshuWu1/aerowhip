# Nominal drone bootstrap fitting

The current job fits the new standalone nominal tracked-origin/attitude response
to the three legacy whips only. It creates a separate candidate and development
assessment. It does not select a GUI model, fit a drone NN, fit the cable, train
PPO, or complete the full M0 bundle.

Current attitude method: [independent_scale_v3 correction](ATTITUDE_MAPPING_CORRECTION_20260908.md).
The controlled comparison freezes every prior translation fit using
`--attitude-only-from <prior-job>`, verifies matching input hashes/reviews,
and fits only the new attitude parameters. Old v1 runs remain immutable.

Implementation: `experimental_data/nominal_pose_fit.py` and
`tools/fit_nominal_drone_pose.py`. Prepared and executed jobs live in
`data/nominal_drone_runs/`. Each job records exact source/input copies, masks,
protocol, optimizer traces, parameters, predictions and preservation checks.

## Fixed data contract

Use phase-aware source `20260908-041031-260837` for whip1_001/002/003.
The user clarified that the supplied OptiTrack files start during takeoff and
end around landing, and answered no to contact/intervention during the CSV
maneuver. File edges are not maneuver edges. Keep all supplied source samples
and all observed native command values; do not append the unexecuted CSV tail.

Initialize approximately 105 ms before CSV onset from the last measured pose
and one second of valid preceding hover. Recompute hover compensation for each
gain candidate. Effective pre-hover frame alignment stays fixed during a
rollout; no future observed pose is a predictor input. Shared geometry is the
archived active offset, not re-estimated inside these drone fits.

Fit position and orientation on the observed CSV maneuver only. Exclude saved
invalid observations, recorded manual exclusions when supplied, position-jump
flags (>10 cm per observed sample; a review heuristic), and uncertain phase
boundary samples. Materialize masks, rather than merely saving review notes.
The current three takes retain 66, 66 and 65 loss samples; each omits one
phase-boundary sample. No extra manual damage interval was invented.

The full 67/67/66 recorded maneuver samples are also included in separate
valid-measurement reporting. The next 0.5 s of logged hold is diagnostic only:
it neither supplies fitting targets nor enters attitude optimizer rollouts.
Its target was selected from the real end position, so that assessment is
conditional on the logged commands, not a preplanned-recovery prediction.
If an extended replay hits the attitude-domain guard, re-evaluate the complete
maneuver alone. Only if that succeeds, report its score and mark the whole
post-hold diagnostic unavailable, with an explicit failure status. Keep the
measured and predicted sample counts separate; do not manufacture recovery
predictions or score the available prefix as a successful full hold. A selected
candidate that fails during the maneuver is rejected as a pose model. The job
continues the remaining development comparisons, reporting its independent
translational subsystem only, with no orientation or attachment prediction.

## Identification order and objective

In the adopted model translation is independent of predicted attitude.
Therefore fit the six translational parameters and effective delay to position
first, then fit the two attitude direction scales and timescale to measured
rotations with those gains/delay fixed. Do not distort translation gains solely to make the
approximate attitude construction match orientation. A poor orientation fit
is a model-design finding to report.

Parameters are shared-horizontal/separate-vertical Kp, Kd and acceleration
feedforward Ga. Position loss is equal-take mean squared XYZ error over the
valid maneuver, normalized by 0.05 m. A fixed weak prior term uses weight 0.01,
center [4,4,3,3,1,1], and scales [20,20,10,10,1,1]. The normalization and prior
are numerical fitting choices, not measured noise/firmware parameters.

Search delay 0–0.12 s in 0.02 s increments. For each delay use three fixed gain
initializations and bounded least squares. Bounds are Kp 0.1–80/s², Kd 0.1–20/s,
and Ga 0–2. Report proximity to bounds, all delay objectives, and the scaled
data-only Jacobian singular values. These are identification bounds, not
verified vehicle limits. Do not infer uniquely identified physical parameters
from a small training error on three similar whips.

Fit s_xy in [0.1,3], s_z in [0.05,1.5] and the critically damped attitude
timescale in [0.02,0.30] s using mean squared geodesic rotation error, with equal
weight per take. The mapping is u=diag(s_xy,s_xy,s_z)*a_O before constructing
the attitude target from u+g and yaw. Evaluate a fixed 27-point starting grid
then refine its three best valid candidates by bounded least squares in log
coordinates. No prior penalty is added to this three-parameter orientation
loss. An entirely rejected search is recorded, never assigned default values.
This is an effective orientation response, not measured actuator
bandwidth or proof of tracking-to-firmware mounting alignment.
Candidates that reach the engine's near-180-degree local attitude-domain guard
are recorded as invalid with their take/reason; they cannot be selected. Other
errors still stop the run. Translation equations and domain guards remain
unchanged; the attitude input mapping is the explicit v3 model revision.

## Numerical evaluation and assessment

The small translational recurrence and its analytic parameter sensitivities
run in NumPy/SciPy on CPU. They use the exact algebra and event partition of
the tested engine's midpoint step, including the gain-dependent hover bias.
Regression tests compare them with that engine and finite differences, and
check recovery of a synthetic noise-free parameter example without shrinkage.
The real fit uses the recorded weak prior.

Attitude optimization uses a cached NumPy midpoint evaluator and SciPy;
full pose/attachment rollouts use Torch/CUDA. Fixed acceleration history is
computed from the predicted translational recurrence, not future measured
position. Scale-dependent effective hover alignment is recomputed each time.
Each fitted result is checked against the CPU fitting evaluator and with maximum
step reduced from 5 ms to 2.5 ms. Save numerical convergence differences; do
not confuse the integration step with command or tracking frequency.

Run a final fit on all three takes and three leave-one-take-out fits. Each fold
selects its gains, delay, attitude scales and timescale using only its two fitting
takes. Its other take supplies only the permissible pre-hover initialization
and known logged command input for prediction, with later measurements reserved
for scoring. Geometry/protocol already informed development, so these remain
development checks, not independent paper evidence.

Report origin position, orientation and attachment position over uninterrupted
maneuver and post-hold segments separately. Compare to the fixed *unfitted
numerical example* with that label; improvement over it is not improvement
over a previous deployed model or a real hit result.

The candidate remains separate until nominal limitations are assessed and the
new drone residual, cable model/residual and consistent 30 Hz policy/export
integration have been completed. Legacy records are still needed for this
bootstrap stage and are not retired by this nominal-only job.
