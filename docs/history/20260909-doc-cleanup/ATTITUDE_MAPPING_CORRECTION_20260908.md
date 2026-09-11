> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Attitude correction and the separate legacy hold diagnostic

The corrected nominal candidate completes all twelve assessment rollouts
(all-three fit plus three development folds, each evaluated on three takes)
without the previous attitude-domain failure. Maneuver orientation errors are
smaller. **This change does not alter the fitted translational dynamics and
does not resolve their post-hold position mismatch.**

Run: `data/nominal_drone_runs/20260908-062555-496491-legacy-whip-nominal-pose`.
Prior comparison: `20260908-060710-079793-legacy-whip-nominal-pose`.
The candidate schema is `nominal_loaded_drone_pose_candidate_v3`, with explicit
`attitude_drive_model=independent_scale_v3`. It is not selected in the existing
GUI/PPO/export path. No NN, cable fitting, training or flight ran.

## The CSV and hold are different phases, correctly represented

The user is correct: the legacy exported sequence contains only the whip.
There is no requirement for its takeoff, pre-hold or post-hold to be CSV rows.
The supplied `Downloads/full_state_pva.py` calls high-level takeoff, streams
the first position through `cmdFullState` for the pre-hold, sends the twenty
maneuver PVA rows, then streams a new constant position through `cmdFullState`
with zero velocity and acceleration. It obtains that post-hold position from
`cf.get_position()` at the end of playback. High-level landing follows.

Thus, in this supplied source, both holds and the maneuver use the FullState
interface, but their setpoints come from different parts of the script. It
does not switch a force controller or change low-level controller parameters
at the maneuver/hold boundary. This source review is not an independent
inspection of every software revision installed on the colleague's computer.

The native logger confirms the relevant inputs:

| Take | Observed maneuver rows | Max difference from supplied script | Post-hold target XYZ (m) | Commanded hold V/A |
|---|---:|---:|---|---|
| whip1_001 | 20 | exactly 0 | [0.59196, 0.00376, 2.32157] | zero / zero |
| whip1_002 | 20 | exactly 0 | [0.59553, 0.00547, 2.18817] | zero / zero |
| whip1_003 | 20 | exactly 0 | [0.57916, 0.01551, 2.24394] | zero / zero |

Each post-hold target is constant in its logged phase. The native post-hold
observation spans are approximately 1.59/1.59/5.09 s; these are logger sample
spans, not exact requested durations. Take 3's longer hold is retained. All
original data and phase labels are unchanged. See `comparison.json` and its
reproduction script `review_comparison.py` in the new run.

The fit still initializes from one second of preceding hover and uses only
66/66/65 maneuver samples in its loss. All 67/67/66 valid maneuver samples
are reported. Only the first 0.5 s of post-hold is replayed as a separate
diagnostic. The predictor receives the **actual logged post-hold command**;
it does not extend the whip CSV or substitute today's full-trajectory export.
Its effective command delay is retained across the transition, so the last
CSV commands can still act in early post-hold.

## Why the attitude mapping changed

The original effective model fitted

    a_O = Kp (p_d - p_O) + Kd (v_d - v_O) + Ga a_d + b

to tracked-origin position, then used `a_O + g e_z` directly to construct
the attitude target. That implicitly equated fitted net motion at the
OptiTrack origin with the acceleration-like quantity that sets controller
thrust direction. Those quantities need not coincide on a loaded drone
whose tracked origin is not its center of mass.

Upstream Mellinger constructs its desired direction from a control target
containing P/V feedback, acceleration feedforward, gravity and integral
compensation. The distinction supports separating the direction mapping;
it does not supply this vehicle's installed gains or prove a fitted empirical
scale is a motor parameter. [Reviewed upstream source, pinned revision](https://github.com/bitcraze/crazyflie-firmware/blob/fe0f5b0ee1c5b9c5e21cb47ad3ae593ddaa86cf7/src/modules/src/controller/controller_mellinger.c#L159-L178).

A preliminary inverse-response-gain factorization removed the singularity
but increased maneuver orientation RMS to approximately 19–20 degrees. It
was not accepted as the fitted correction. The new model instead identifies
a small independent direction mapping from measured orientations:

    u = diag(s_xy, s_xy, s_z) a_O
    z_C = normalize(u + g e_z)
    y_C = normalize(z_C cross [cos(yaw), sin(yaw), 0])
    x_C = y_C cross z_C
    R_target = [x_C, y_C, z_C] R_CT

The existing second-order SO(3) response follows this target using its fitted
timescale tau. The two positive scales are **empirical nominal attitude
parameters**, not an NN, verified thrust calibration, identified cable load
or proof of physical feasibility. This low-parameter mapping remains an
approximation; future data must assess its generalization.

The R_CT effective pre-hover alignment is recomputed consistently for each
scale candidate using the same fixed past hover observations. Future measured
orientation is never used during the rollout. The SO(3) guard and singular
heading/direction checks remain in place. No tilt/acceleration clipping or
replacement of failed samples was introduced.

## Controlled fitting and results

Every final/fold translational parameter and delay is frozen from the matching
previous fit. Exact copies and hashes live under `inputs/frozen_translation`.
The data version, input file hashes and reviews must match before reuse.
Only s_xy, s_z and tau are fitted, using equal-take geodesic orientation loss
on the maneuver. The search evaluates a fixed 27-point initialization grid,
then refines from its three best valid starts in log coordinates. Bounds are
s_xy=[0.1,3], s_z=[0.05,1.5], tau=[0.02,0.30] s. They are numerical identification
bounds, not measured aircraft limits. Post-hold targets do not enter this loss.

Final s_xy=1.00463, s_z=0.463971, tau=0.021641 s.

| Take | Old final orientation RMS | New final orientation RMS | Old left-out orientation RMS | New left-out orientation RMS | New final attachment RMS |
|---|---:|---:|---:|---:|---:|
| whip1_001 | 5.40° | 2.16° | unavailable: rejected pose fit | 3.44° | 1.95 cm |
| whip1_002 | 5.16° | 2.17° | 5.49° | 2.45° | 2.28 cm |
| whip1_003 | 4.95° | 2.20° | 4.85° | 1.72° | 2.81 cm |

The final tracked-origin position RMS remains 1.88/2.22/2.73 cm. Its previously
available predictions are bit-identical. The previously rejected fold used
CPU translation-only output, and comparison to its new GPU output differs by
less than 9e-15 m. This is round-off, not a changed trajectory.

All twelve full diagnostic pose predictions are now finite. For the final
candidate, sampled target tilt remains below 61 degrees, its minimum vertical
direction component before normalization is about 4.49 m/s², and adjacent
sampled target-frame changes stay below 43 degrees instead of approaching
180 degrees. These are sampled model diagnostics, not certified flight limits.

Tau lies near the 20 ms search floor for the final model and two folds; the
fold excluding take 1 reaches that floor. Do not report a precisely identified
actuator bandwidth from 100 Hz pose data. All gain, optimizer and bound records
are retained. These three similar legacy takes have already informed model
design, so the leave-one-out results remain development evidence.

## What “post-hold prediction is poor” means

After the observed hold command begins, the model and real drone brake/move
differently even though the replay supplies the logged command. A command of
zero velocity does not mean the real drone's velocity instantly becomes zero.
The model must still predict the transient from the incoming state.

Final origin RMS over the next 0.5 s is 15.16/18.49/14.76 cm. Final orientation
RMS is 13.77/14.44/13.56 degrees; all three can now be assessed. The fit did not
use this regime, and its frozen hover compensation/effective dynamics do not
establish an accurate braking model. Errors in end velocity, timing, controller
memory or loaded response could contribute; this run does not identify one
of those as the measured cause.

This is **not evidence that the recordings used the wrong CSV**, and not a
reason to discard the holds or require old takeoff/landing to have been in
the maneuver CSV. It is a separate limitation of extrapolating this short
maneuver-fit response into the scripted hold. Because translation is currently
independent of predicted attitude, an attitude-only correction cannot change
that position error. The hold diagnostic also uses a real-end-selected target,
so it is not validation of the future preplanned gentle recovery export.

The observed attitude failure is resolved on these recordings. Drone residual,
cable physics/residual, combined validation and the new 30 Hz PPO integration
are still later work. No complete M0 or hover-to-hover model is declared ready
from this result.

## Tests and artifacts

**69 targeted tests passed on Windows 11 / NVIDIA RTX 4080.** They cover
independent adaptive-ODE comparison, fitted scales and their gradients,
causal scale-dependent hover alignment, synthetic parameter recovery, phase
handling and unchanged domain guards. A cached NumPy/SciPy evaluator makes
the small attitude fit inexpensive; each candidate is checked against full
Torch/CUDA pose prediction. Their rotation matrices agree within 1e-10.
Halving the integration step changes orientation by at most 0.20 degrees and
attachment position by less than 0.20 mm across these assessments.

The run preserves protocol/source/input copies, masks, frozen translation,
attitude searches, all fold predictions, CPU/GPU parity, timestep convergence,
transition audit and comparison data. Plot (historical artifact, no longer included in this checkout).
All 133 protected raw/model/policy/config files remain unchanged. The old
candidate and its failed checks remain available. PPO and SAC stay stopped.
