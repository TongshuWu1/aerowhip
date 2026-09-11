# Full drone and cable adaptation

User-authorized implementation and measured-data run, 10 September 2026.
Current job: `runs/adaptation/M1-full-whip-v2`. Read its `status.json` before doing any
work. Do not duplicate or automatically restart it. The earlier gain-only M1
remains separate evidence; this candidate is named **M1-full**, also a child of M0.

**Completed and registered, not promoted.** Both residual stages stopped by
practical plateau (drone 585 total updates, cable 280) and passed their selected
checkpoint gradient checks. Excluded whip 005 improves from 8.93 to 7.64 cm drone
RMS and 14.89 to 8.81 cm command-driven tip RMS. Take 004's tip error worsens
10.40 to 10.98 cm. M0 remains selected; there is no physical M1 flight.

The v1 attempt is preserved after failing the cable NN gradient check at its
final physical iterate, before any cable NN training. V2 resumes the saved drone
optimizer at update 400, removes the routine neural ceiling, and uses the best
numerically verified physical iterate. Read [the detailed review](../development/TRAINING_HORIZON_RESIDUAL_REVIEW.md).
No integration substeps, smoothing, data roles or training-window lengths changed.
The separately frozen continuation is implemented by `whip_full_continuation.py`
and `tools/continue_full_whip.py`; do not invoke the v1 fitter on its v2 protocol.

## Scope and exact experiment

This workflow updates the reusable forward model between flights. It contains
all four learning blocks: nominal drone response, drone residual, nominal cable
physics, and cable residual. It is implemented in `experimental_data/whip_full_*`
and exposed by `tools/adapt_whip.py prepare-full` and `fit-full`.

The source is the selected repaired-system M0 that generated the original MPPI
flight. No old-system model, archived neural checkpoint, changed reward or target
is used as a substitute. The commanded and measured coordinates remain global.
The 145 g drone, 17 g cable assembly, marker geometry, rotated attachment offset,
numerical model and time step remain bound to M0. No generated-command acceleration
cap is introduced. Effective response parameters are not measured motor limits.

The flown 309-row CSV and its original forecast remain immutable. Training and
same-flight comparisons score the **1.1333 s whip segment**, not the subsequent
recovery or the complete 10.2667 s command file. These reinitialized predictions
are distinct from the saved original forecast.

## Data preparation and separation

- Whip 001/002/004 are adaptation. 005 is excluded from every optimizer, delay
  choice and checkpoint selection. Its earlier inspection during method review
  is disclosed; this is not a new blind test. 003 stays reserved and excluded
  from the strict cable initializer because of missing c5 observations.
- All 69 eligible two-second preliminary **training** drone windows are retained,
  giving 72 drone windows with the three complete whip segments. Cable fitting
  uses two one-second windows per preliminary window where the causal history
  and masks permit, plus the three whip segments: 135 accepted, six excluded.
  `cable_window_review.json` records each exclusion and its reason.
- Original preliminary validation take `figure8_002` is copied into a separate
  validation-window list and evaluated only after candidate selection is frozen.
  The training loader refuses validation roles. Neither archived different-system
  recordings nor validation frames enter the training replay.
- The preparation verifies original raw hashes, prepared arrays and the parent
  source snapshot, then makes a separately hashed input and code snapshot. It
  does not recenter XYZ, renormalize height, realign commands to desired motion,
  fill missing cable observations or alter whole-take roles.
- Initialization uses 0.4 s of past drone measurements and one second of past
  cable observations, with the existing 0.02 s cable endpoint-velocity weighting.
  The current effective compensation/alignment are nuisance initial states, not
  measured onboard integral state or calibrated mounting orientation.

Whip and preliminary training family masses are 1 and 0.5, normalized to 2/3 and
1/3. Takes receive equal weight within each family; windows share their take's
weight. More windows or native samples cannot silently increase a take's weight.

## Optimization and model meaning

1. **Nominal drone translation:** optimize all six positive XY/Z feedback,
   velocity-feedback and acceleration-feedforward coefficients. Recompute their
   dependent causal compensation consistently. The inherited NN is fixed here.
   Repeat this small parameter fit at each declared effective delay: 0, 10, 20,
   30, 40, 60, 80, 100 and 120 ms. The timing is an effective closed-loop delay
   confounded with estimated stream alignment, not identified actuator latency.
2. **Nominal attitude:** optimize XY/Z acceleration-to-attitude scales and attitude
   response time constant using native measured orientation and the production
   SO(3) midpoint equations. Translation and attitude are separate response blocks.
3. **Drone NN:** freeze nominal translation and fine-tune the inherited width-16
   network through complete recursive position rollouts. Penalize correction
   magnitude and changes from the inherited NN evaluated at the same predicted
   state. Its existing per-axis acceleration correction bound is ±0.5 m/s².
   Refine nominal attitude afterward because the changed translation affects its
   feedback-driven desired attitude.
4. **Nominal cable:** freeze the adapted drone. Fit EI, bending damping Cb and
   external velocity damping together, using measured rotated attachment motion.
   This prevents drone boundary error from being assigned to cable parameters.
   Coefficients are effective fitted parameters; sensitivity or a bound-active
   result does not establish precise material identification.
5. **Cable NN:** freeze cable physics. Introduce an explicitly zero-output width-32
   acceleration residual, then train it through complete cable rollouts. It uses
   relative cable geometry/velocities and root velocity, with no time, target or
   take identity. Each free-node axis has a ±0.5 m/s² correction bound; root
   acceleration correction is zero. It adds no second learned damping term.
6. **Combined validation:** save the complete candidate and immutable selection
   record first. Then evaluate recorded PVA through updated drone → rotated
   attachment → cable without supplying future measured attachment positions.
   Also retain conditional cable diagnostics, per-axis drone/attitude errors and
   the original preliminary holdout check.

The current forward model is an empirical loaded-drone cascade. It does not feed
explicit simulated cable tension back into the drone. The residual is model
prediction discrepancy, not target distance or desired-minus-measured position
copied directly into a force. There is no hit reward in adaptation.

For a vector error e and declared scale s, the trajectory penalty is
`2*(sqrt(1 + ||e/s||²) - 1)`. Position and cable scales are 0.02 m; orientation
scale is 0.05 rad. These are objective scales, not calibrated noise variances.
Cable loss gives half its observation mass to all markers and half to the tip,
under fixed missingness masks. Padding has zero score and zero loss gradient.
Nominal fits add `0.03*||log(theta/theta_parent)||²`; effective-delay prior is
`0.03*((delay-delay_parent)/0.04)²`. Both residual penalties have weight 0.01.

The exact bounds, seed, weighting and stopping settings are in the frozen
`protocol.json/full_update`. Bounds on model parameters do not clip commanded
trajectory acceleration. Parameter stages use bounded trust-region least squares
with GPU batched differences; residual stages use Adam, clipping gradient norm
at 1.0. This clips optimizer updates, not drone commands.

Best and current NN states, Adam state, parameter-search best iterates, histories
and termination reasons are preserved. Practical plateau, solver termination,
manual stop and safety ceiling are distinct. Reaching a ceiling is not convergence.
No validation metric selects a parameter or checkpoint, and model registration
does not promote or select the candidate for flight.

## Numerical checks and efficiency

Windows and finite-difference candidates run in float64 batches on the RTX 4080.
GPU graphs cover response rollouts and cable forward/VJP blocks. Cable gradients
propagate through the entire scored window, with recomputation inside captured
blocks rather than detached recurrent states. No data reduction is used to make
the gradient check pass.

The representative mixed-duration check matched production drone position within
0.0069 mm and orientation matrix entries within 4.1e-5. Cable physical forward and
zero-output cable residual predictions matched exactly in the checked runtime.

The full 135-window check found that a 1e-4 log-parameter difference was too coarse:
Cb derivatives differed by about 47.7% at half that step. A resolution study on
the unchanged physics/data showed convergence near 1e-6, with 0.458% discrepancy
at half the step. The full cable NN derivative also needed smaller perturbations;
at 1e-7, its checked derivative agreed with backpropagation within 0.024%, and the
neighboring checked step agreed within the declared 2% tolerance. Every attempted
step is recorded, including failed coarse checks. This is a local derivative
check, not proof of globally well-conditioned optimization.

`runs/audits/full-adaptation-20260910` preserves smoke fixtures, full-batch results,
the resolution study and source scripts. Audit fixtures are not fitted M1 evidence.
The full-batch check used under 0.5 GiB of PyTorch reserved GPU memory; performance
claims apply to native Windows/RTX 4080 only.

## Reading results and the next loop

| Excluded-data metric | M0 | M1-full |
|---|---:|---:|
| Whip 005 drone RMS [cm] | 8.9299 | 7.6352 |
| Whip 005 cable tip, measured attachment [cm] | 6.7405 | 3.9527 |
| Whip 005 cable tip, command-driven [cm] | 14.8928 | 8.8111 |
| Preliminary holdout mean 2 s drone RMS [cm], 26 windows | 9.8075 | 7.7974 |
| Preliminary holdout mean 1 s conditional tip RMS [cm], 42 windows | 4.9401 | 4.9231 |

The full candidate improves the excluded whip's coupled prediction, but the
individual residual results are mixed. At fixed final nominal parameters, removing
the drone NN improves whip 005 drone RMS to 6.2633 cm and coupled tip to 8.2092 cm,
while worsening preliminary holdout drone RMS to 12.7054 cm. Adding the cable NN
improves 005 conditional tip from 6.5799 to 3.9527 cm and coupled tip (drone NN on)
from 11.5659 to 8.8111 cm. It slightly worsens preliminary conditional tip relative
to the updated physical-only stage, 4.8207 to 4.9231 cm. Do not claim universal
benefit or necessity of both networks. These are fixed-parameter removals, not
independently refitted non-neural alternatives. See the detailed review for all
four residual combinations and the limits audit.

V2 continuation plus validation took 1950.1 s (32.5 min) on Windows/RTX 4080;
the cable NN stage took 1718.7 s. This excludes earlier V1 fitting and numerical
investigation. All 135 cable windows were retained. Thirty-eight targeted tests,
local GPU derivative/production checks and frozen source/data/model/flight hash
checks passed. GPU utilization reached 100%; that is utilization evidence, not a
proof of optimal hardware throughput.

Use **Flight comparison → Compare generations** for saved same-input model
predictions. Select the full job in its fitting-progress view to see stage status
and loss. M1-full is a sibling of the earlier partial M1, not an M2 flight.

The complete model is saved under `candidate/model.json`, intermediate models
under `stages`, selection under `fit/selection_frozen.json`, and the final combined
comparison under `runs/evaluation/M1-full-whip-v2`. Consult `status.json` and
`fit/result.json`; file presence alone is not evidence of completed fitting.
The separate `runs/evaluation/M1-full-whip-components` report is available in the
same UI for stage and residual-removal inspection. Removal variants are audit-only,
not registered or selected models. The lineage card shows actual parent edges,
and the comparison plot retains enough height for its axes and labels.

Next prospective evidence requires a new plan and frozen forecast from the
reviewed candidate, followed by a later real flight. An improved retrospective
prediction does not itself establish a better physical strike. M0's original
forecast must never be regenerated with M1 and relabeled as the original.

The staged approach follows the research rationale and primary literature in
[the model contract](ADAPTATION_MODEL_CONTRACT.md#connection-to-primary-literature).
The particular stage ordering, objective scales and replay weighting are project
design choices, not a claim to reproduce a paper exactly.
