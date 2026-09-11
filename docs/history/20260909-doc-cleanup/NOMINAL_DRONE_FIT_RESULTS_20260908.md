> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# First nominal drone pose fit: useful position fit, unresolved attitude model

The nominal-only bootstrap fit is complete. It reproduces the observed legacy
whip position transient to a few centimeters, but **the pose model is not ready
for PPO or a complete M0 bundle**. One development fold cannot produce a valid
attitude fit; the final model also fails its extended attitude replay on take 3
and predicts the other post-holds poorly. No drone NN, cable fit or PPO ran.

Run: `data/nominal_drone_runs/20260908-060710-079793-legacy-whip-nominal-pose`.
Status: `NOMINAL_FIT_ASSESSED_MODEL_LIMITATIONS`, `deployment_ready=false`.
Protocol: [nominal fitting](NOMINAL_DRONE_FITTING.md).
Plot: measured and predicted trajectories (historical artifact, no longer included in this checkout).

## What was fitted and what data were used

Only whip1_001, whip1_002 and whip1_003 from the phase-aware processing version
`20260908-041031-260837`. Preliminary figure-eight/oscillation recordings did
not enter this drone fit; they remain eligible for later cable fitting.

The user clarified that these supplied OptiTrack clips start during takeoff
and finish around landing, and answered no to contact/intervention during the
CSV maneuver. File boundaries are not maneuver boundaries. Supplied raw and
processed files were preserved; the job has its own exact input copies.

The loss uses 66/66/65 valid maneuver pose samples. One uncertain onset-boundary
sample per take is excluded from the loss, but all 67/67/66 valid recorded
maneuver samples are included in the reported errors. No additional manual
exclusion or jump interval was needed. Bad predictions were not used as a
reason to discard measurements. Saved review intervals and exclusions are
materialized by this fitter; this does not retrofit all older UI/loaders.

Initialize from a fixed one-second pre-hover history, ending approximately
105 ms before the first observed CSV command. Recompute the hover compensation
at every gain candidate. Predict continuously from that initial state using
the native logged P/V/A commands, with no later measured-state resets. The
next 0.5 s of logged hold is a separate diagnostic, never part of the loss.
Its target was chosen from the real end position, so it is a conditional
logged-command replay, not an autonomous preplanned-recovery validation.

## Maneuver results

Errors below are RMS Euclidean position error and RMS geodesic orientation
error. They are not cable-tip error, hit error, or accuracy at the unexecuted
0.78 s planned hit. Only 20 CSV rows were observed; the missing tail was not
invented.

| Take | Final all-three origin RMS | Final attachment RMS | Final orientation RMS | Left-out origin RMS | Left-out attachment RMS | Left-out orientation RMS |
|---|---:|---:|---:|---:|---:|---:|
| whip1_001 | 1.88 cm | 2.07 cm | 5.40° | 2.59 cm | unavailable | unavailable |
| whip1_002 | 2.22 cm | 2.41 cm | 5.16° | 2.55 cm | 2.76 cm | 5.49° |
| whip1_003 | 2.73 cm | 2.85 cm | 4.95° | 3.59 cm | 3.66 cm | 4.85° |

The fold excluding take 1 has a usable independent translational fit, but
its attitude search is rejected on a fitting take at every point of the
fixed eleven-point timescale grid. It is a rejected pose candidate. No
default attitude, carried-over weights or measured future orientation fill
that gap. Its rotation and attachment outputs are explicitly unavailable;
the absent orientation curve in the plot is not zero error.

These are **development comparisons**. The three similar whips, shared
geometry and protocol have already informed development. They do not provide
independent experimental or paper evidence. The saved unfitted-example
comparison is a numerical reference, not the previously deployed fitted model.

## Fitted effective response parameters

The final all-three candidate has:

| Parameter | Value |
|---|---:|
| Kp horizontal / vertical | 5.9278 / 21.4053 s⁻² |
| Kd horizontal / vertical | 6.1531 / 1.8240 s⁻¹ |
| Acceleration feedforward horizontal / vertical | 0.3041 / 1.9075 |
| Effective delay | 0.060 s |
| Attitude response timescale | 0.02966 s |

These are fitted effective tracking parameters, not firmware controller gains,
motor constants or a measured communications delay. Timing alignment and
cached logger observations can contribute to the estimated delay. The delay
search used 20 ms increments; it does not establish 1 ms precision.

The final scaled data-Jacobian condition number is 34.28. More importantly,
the fold excluding take 3 puts horizontal Kp at its 0.1 s⁻² lower identification
bound, whereas the other folds give 7.42 and 11.45. Position accuracy across
similar short trajectories does not uniquely establish the parameter values.
Do not widen the bounds merely to improve a score.

## The limitation revealed by fitting

The nominal translation equation is an effective P/V/A tracking response.
Its modeled tracked-origin acceleration is then reused as `a_O + g e_z` to
construct the attitude target. The fit currently optimizes position before
the attitude timescale, with no attitude/command-feasibility constraint on
the translational gains.

For the final model, sampled maneuver acceleration reaches approximately
−11.46 m/s² vertically in takes 1 and 2. The effective command frame consequently
tilts beyond 101° and changes by approximately 179° between adjacent sampled
times. The guard is retained: it rejects local attitude errors too close to
180°. Nearby fits can therefore give small position errors but an unusable
attitude rollout. Changing tau alone did not resolve the rejected fold.

These values come from the **model**, not measured motor thrust or proof that
the real drone inverted. O is a non-COM tracking origin, and the loaded cable
also matters. `a_O + g` was always an approximation to an attitude-driving
direction, not an identified thrust plant. The additional diagnostics save
native delayed command samples, modeled acceleration and the constructed
frame separately; their extrema are at measurement times, not all substeps.

Post-hold performance confirms the limitation. Final origin RMS over the
0.5 s diagnostic is **15.16 cm and 18.49 cm** on takes 1 and 2; orientation
RMS is **29.73° and 24.25°**. Take 3 encounters the local attitude-domain guard.
Its entire post-hold pose diagnostic is marked unavailable, not truncated
and reported as successful. Full measured sample counts remain recorded.
Some delayed last-maneuver commands still act during early post-hold; the
phase label describes logged playback time, not an instantaneous plant switch.

## What should happen next

Keep this translation fit as a diagnostic starting point. Resolve the nominal
attitude-target construction and its admissible command domain before learning
an NN on top of it or starting new PPO. The correction must distinguish
effective tracked-origin acceleration from the attitude-driving controller
quantity; silently clipping reported acceleration or loosening the SO(3)
guard would not fix that interpretation. Any chosen limits must be explicit
model assumptions or supported vehicle/interface values, not invented hardware
specifications.

Then repeat this same nominal assessment on the preserved inputs. Once the
nominal domain is satisfactory, fit the small drone residual to remaining
systematic error, fit cable physics/residual using measured attachment motion,
and assess the combined recursive prediction. The new 30 Hz force PPO and
30 Hz export integration remain subsequent work. The current candidate is
not selected in the GUI or substituted into the existing export path.

## Reproducibility and preservation

The run stores protocol/source/input hashes, per-take masks, all gain starts
and delay profiles, attitude grid/refinement traces including rejected
candidates, candidate or rejection records, predictions, scores, plots and
additional acceleration diagnostics. Earlier interrupted fitting jobs are
preserved with failure status; no failed candidate was deleted or relabelled.

**62 targeted tests passed on Windows 11 / NVIDIA RTX 4080.** They cover the
model/initializer/data adapter, legacy command phases, analytic fitting
sensitivities, synthetic recovery, exclusions, and explicit domain-failure
reporting. Valid GPU pose rollouts agree with the CPU translational evaluator
within 1e-14 m. Halving the integration step from 5 ms to 2.5 ms changes the
reported position predictions by at most 0.034 mm across the assessments.
Orientation/attachment convergence is saved where those outputs exist; this
is numerical evidence, not a hardware tracking guarantee.

All 133 protected original/model/policy/configuration files were verified
unchanged. PPO and SAC remain stopped. No residual fit, cable fit, active
calibration change, data retirement or flight occurred. Legacy records must
remain available until the complete preliminary bundle is established.
