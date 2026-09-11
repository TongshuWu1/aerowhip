> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Nominal drone implementation review — 8 September 2026

The new standalone engine implements the stated effective loaded-drone model
consistently. This review found and fixed data-contract and numerical-check
gaps. It does **not** establish prediction accuracy, a fitted preliminary
baseline, or completion of the new PPO/export pipeline.

Review scope: `drone_pose_response.py`, its causal initializer and phase-aware
data adapter, geometry, regression tests, and the existing training/export
interfaces. The nominal equations, active physical calibration, selected policy,
raw measurements, processed snapshots and residual checkpoints were preserved.
No fitting, PPO launch, controller change or flight command occurred.

## Intended pipeline and current status

1. A future new **30 Hz force policy** generates a frozen virtual force sequence.
2. The mechanically coupled virtual point-mass/cable simulation generates the
   desired reference. Convert that reference to the tracked drone origin and
   sample consistent P/V/A at 30 Hz.
3. The effective drone response predicts actual origin P/V and orientation from
   the initial measured hover state and the desired FullState commands.
4. Exact rigid-offset kinematics produce the predicted cable attachment motion.
5. Cable physics plus its residual predict cable motion from that boundary.
6. PPO evaluates predicted executed cable motion. Export the desired reference
   from step 2, so the real controller receives the same modeled input.

Steps 3–4 now have a tested **nominal, unfitted standalone implementation**.
There is no drone NN in this new engine yet. Existing drone residual weights
belong to an older model and must not be relabelled as a fit of this engine.
The existing PPO and Full State UI still use their historical execution path;
they have not been migrated to the new model or new 30 Hz force policy.

## Findings fixed in this review

| Finding | Consequence | Correction |
|---|---|---|
| Finite pose samples could bypass saved validity masks | A masked hover sample could affect initial state/compensation; orientation/attachment target masks could be wrong | Honor position and orientation masks, including derived attachment validity |
| A requested hover window could be silently shortened | Compensation could use less history than the recorded protocol requested | Require the requested duration, allowing one observed sampling interval at the endpoint |
| Maneuver-only selection required its final tracking timestamp to reach the command boundary | Valid data were rejected when command and tracking clocks did not share an exact sample | Check coverage against the recording; retain measured sample times without inventing a terminal observation |
| Named command columns were not checked | An incompatible column layout could silently change P/V/A meaning | Verify exact saved P/V/A/yaw/rate column names/order |
| A NaN command-query timestamp returned a cached command | An invalid public API query could produce a seemingly valid input | Reject nonfinite query timestamps |
| Timestep check considered only attitude dynamics | Large translational gains could remain numerically unresolved | Add an explicit gain-dependent translational resolution check; retain convergence testing |
| Phase-boundary/coverage details were absent from adapter outputs | Later scoring could lose uncertainty and silently treat partial post-hold as complete | Return boundary/command validity, CSV identity, requested/observed end times and coverage metadata |

Regression cases reproduced mask, duration, maneuver-only and NaN issues before
the fixes. Checks also cover column-order rejection and the new resolution rule.
The fixed numerical example predictions for all three real takes remain
bit-identical to the previous probe under the same input history and gains.

## Checks of equations and computation

- Tracked origin O and cable attachment A remain different reference points.
  `p_A=p_O+Rr`, `v_A=v_O+R(omega cross r)` and both angular-acceleration and
  centripetal terms in attachment acceleration are included. The 55 mm rigid
  vertical offset and separate 63 mm flexible first cable span are not merged.
- Effective translational acceleration receives no extra gravity or cable
  reaction. The loaded response is not fed the virtual force as motor thrust.
  The constant hover compensation has the sign and units required by the mean
  observed PD balance.
- Attitude uses an explicit tracked-frame alignment and measured initial angular
  velocity. Multi-axis integration agrees with a separately implemented SciPy
  DOP853 solve and converges at second order as the timestep is reduced.
  Independent analytic translation and single-axis attitude solutions also pass.
- Delayed zero-order-held commands are resolved at their actual event boundaries,
  including exact 1/30 s planned events. Native legacy timing remains first
  logger observation, not verified vehicle actuation time. Missing or expired
  command intervals are rejected.
- Future measured poses are diagnostic targets only. Changing them cannot change
  the rollout. Gradient checks include re-estimation of gain-dependent hover
  compensation from the same past history, on CPU and CUDA.
- Distinct batch members agree with analytic translation and CPU/CUDA results
  in float32 and float64. These checks supplement the original identical-batch
  test and the independent frame/attachment kinematics checks.

## Modeling assumptions that still need assessment

This is an effective response of the same installed drone/cable setup. Its
translation is not dynamically constrained by its predicted tilt through an
identified thrust plant. Consequently, it must not be presented as a physically
complete two-way drone/cable model or used as a feasibility certificate.

In particular, `a_O+g e_z` is not an exact expression for motor-thrust direction
when cable reaction and origin-to-COM motion matter. The pre-hover alignment can
absorb steady tilt and tracking-frame mounting but cannot remove time-varying
cable effects. The upcoming nominal assessment must inspect orientation and
attachment errors alongside origin-position error. A low position loss alone
does not validate the cable boundary prediction.

Hover compensation is frozen, not measured controller integral. Long recovery
behavior is outside the current assessment. The legacy post-hold target was
selected from the measured end position; replay is conditional on that logged
input and must be reported separately from the maneuver. The future fully
preplanned recovery is a different command-generation procedure.

Only three similar whips are eligible for drone identification. Shared XY
parameters reduce freedom but do not make weakly excited parameters identifiable.
Do not give each take a freely fitted bias or reuse precomputed compensation as
gains change. Clock/log latency and effective command delay remain confounded;
a boundary-optimal delay is a diagnostic, not proof of hardware latency.

The cable candidate remains constrained dissipative correction, not an arbitrary
missing-force model. Its separate external drag must remain zero in the new
NN-only fit. Existing active calibration and historical candidates were not
silently changed during this review.

## Remaining work before a new policy

Fit and assess nominal origin **and orientation** response on complete observed
whip segments, with separate hold scores and parameter/timestep sensitivity.
Then fit the new drone residual against the remaining error and combine it
with the separately assessed cable model/residual. Preserve development versus
future prospective assessment distinctions.

The later PPO/export migration must also address the previously recorded
feasibility gate, post-hit numerical-failure accounting, initial cable
information mismatch, interpolation acceleration ripple, residual packaging,
and virtual-preview versus predicted-execution mismatch. These older issues
are listed in `COMPLETE_MODEL_AUDIT_20260908.md`; this standalone-engine review
does not claim to have fixed them.

Evidence: `runs/audits/20260908-050940-048699-nominal-pose-implementation-review/`.
Final result: **59 tests passed**, all three recorded-take replay checks passed,
and 133 protected files were verified unchanged. Exact results, hardware, source
copies and preservation checks are recorded there. Testing is local
Windows/RTX 4080 only, not Ubuntu or a flown validation.
