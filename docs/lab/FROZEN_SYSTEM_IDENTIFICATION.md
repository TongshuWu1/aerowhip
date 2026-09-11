# Frozen staged system identification, version 1

Frozen for the user-authorized M2 development refit on 10 September 2026.
The method is **staged, regularized nonlinear system identification by
simulation-error minimization**. The numerical optimizers are bounded nonlinear
least squares for physical/response parameters and Adam for neural residuals.
This is a supporting method for the UAV real-to-sim-to-real whipping system,
not a claim of a new adaptation algorithm.

The executable contract, source snapshots, input hashes, environment and this
document's frozen copy are in
`runs/audits/M2-frozen-refit-v1/`. The new job and model ID are
`M2-frozen-refit-v1`; its parent is `M1-full`, generation 1. The new candidate is
generation 2. It starts from the same M1 as `M2-full-whip-v1`, not from that M2.
The earlier M2 and its evidence remain intact. No automatic promotion or MPPI
run follows the refit.

## What is fixed

This freezes the existing staged implementation. The combined command-to-cable
fitting objective in `SYSTEMATIC_ADAPTATION_PROTOCOL.md` remains a separate,
unimplemented proposal. Here, combined prediction is evaluated after selection;
it is not an optimization term. Repeating the same parent, data and method may
reproduce M2's mixed combined prediction. An improvement is not guaranteed.

| Item | Frozen rule |
|---|---|
| New M1 flights | 001/002/004 for fitting; 003/005 for post-selection evaluation |
| Prior M0 flights | Only prior adaptation takes 001/002/004 enter replay |
| Preliminary data | Existing training windows; figure8_002 remains held out |
| Family weights | New whip 0.50, prior whip 0.25, preliminary 0.25 |
| Within families | Equal take weight, shared over eligible windows within each take |
| Fitting interval | Full available planned whip interval (1.2 s for this batch); existing preliminary windows |
| Initialization | Causal cable history 1 s, drone history 0.4 s; no future-state resets during scored rollouts |
| Coordinates | Raw global tracking frame, measured orientation and rotated attachment offset |
| Observations | Existing reviewed masks and timestamps; no fabricated markers or height normalization |
| Warm start | Parent nominal parameters and both parent residual networks; fresh Adam state |
| Hardware geometry | 145 g drone, 17 g cable assembly; measured geometry/offset unchanged |
| Random seed | 20260910; record runtime/source versions; bitwise cross-device reproducibility is not assumed |

The frozen preparation retains the original raw hashes and reviewed arrays. Clock
alignment is estimated; transport delay is not independently measured. A fitted
delay is an effective response parameter. The current drone model describes the
loaded PVA response; it does not identify a motor thrust ceiling or include an
additional explicit cable-reaction force.

## Fitting objective and order

For a measured trajectory, roll the model forward recursively from its causal
initial state using the logged command schedule. Minimize weighted error between
simulated and measured trajectories. Position and cable errors use a 0.02 m scale;
orientation uses 0.05 rad. The robust penalty on a squared normalized vector error
is `rho(s) = 2 * (sqrt(1+s) - 1)`. These are fixed engineering scales, not measured
sensor standard deviations. Use valid observations only, with normalized weights.

1. **Drone nominal response:** fit `kp_xy`, `kp_z`, `kd_xy`, `kd_z`,
   `feedforward_xy`, `feedforward_z` to position rollouts, retaining the inherited
   residual. Profile the fixed delays
   `[0, .01, .02, .03, .04, .06, .08, .10, .12]` seconds using training loss and
   the delay prior. Fit the attitude response separately to orientation.
2. **Drone residual:** fit the inherited neural acceleration correction by
   differentiating the entire drone rollout. Penalize correction magnitude and
   change from the parent's output at the same simulated states. Refine the
   attitude parameters after this stage.
3. **Cable physics:** fit positive `EI`, `Cb` and external drag with cable motion
   driven by the measured attachment trajectory. The marker objective averages
   0.5 all-marker loss and 0.5 tip loss. Keep the inherited cable residual fixed.
4. **Cable residual:** fit the inherited neural correction through the entire
   cable rollout, with the same marker objective and residual regularization.
5. **Freeze selection, then evaluate:** save the selected parameters and weights
   before any validation. Evaluate drone, measured-attachment cable, and combined
   command-to-tip predictions on every eligible take; separately report new
   validation and prior/preliminary retention.

Positive nominal parameters use log coordinates. The parent-parameter penalty
coefficient is 0.03; the delay prior is
`0.03 * ((delay - parent_delay) / 0.04)^2`. Both residual magnitude and
parent-output-change coefficients are 0.01, normalized by the correction scale
squared. Residual corrections retain the existing 0.5 m/s² bound. This is model
regularization, not a measured aircraft acceleration limit. No command speed or
acceleration limit is changed by this refit. The executable contract records every
parameter bound and scale.

## Optimization and numerical checks

Physical/response stages use SciPy `least_squares`, trust-region reflective (`trf`),
with bounds and `ftol = xtol = gtol = 1e-6`. Central finite differences are batched
on CUDA: log step `1e-4` for drone/attitude and `1e-6` for cable parameters.
The outer least-squares solver runs on CPU; expensive rollouts run on GPU.

Neural stages use Adam: learning rate 0.001, weight decay 0.0001, gradient norm
clip 1, full temporal gradients. CUDA graphs/batched trials reduce launch overhead;
cable blocks retain gradient propagation across the complete fitting window.
The checked runtime is Windows, RTX 4080, CUDA, float64. Save current/best weights,
optimizer state, loss history and actual stopping reason.

| Stage | Practical stopping | Numerical ceiling |
|---|---|---|
| Least squares | Minimum 6 callback updates; patience 5; relative improvement 0.001; solver tolerances may also stop it | 80 function evaluations per solve; report a ceiling as a ceiling |
| Each residual | Check every 5 updates; minimum 40; patience 6 checks; relative improvement 0.005 | None |

Check full-rollout residual gradients against finite differences before training
and for the selected checkpoint. Selection uses the best numerically valid
training checkpoint; a failed numerical check is recorded. Validation error never
chooses an iterate, stopping time, parameter bound or residual checkpoint.

## Interpretation and future repetitions

Report both each take and equal-take means. Component improvement alone does not
establish better combined prediction or a stronger real strike. Existing held-out
takes were inspected during development; they are not an untouched final paper
test. Current results remain development evidence.

At later generations, reuse this stage order, scales, bounds, stopping and
initialization, with the new parent and predeclared new training takes. In the
current implementation all older eligible whip takes share one family, with equal
take weights; equal weighting across older generations is not separately
implemented. Normalize family masses 1 (new), 0.5 (all prior whip), 0.5
(preliminary) over families that exist. Changing this rule, the model class or
adding combined fitting requires a named new method version rather than silent
per-round tuning.

Before the clean paper experiment, freeze the chosen method, MPPI settings and
evaluation protocol together. Fit a fresh M0 from new preliminary recordings
with reset learned weights. Preserve current development artifacts separately.
