# AeroWhip: one-day experiment protocol

Updated 11 September 2026. This is the user's current minimal study: retain the
existing M0 and preliminary data, then collect fresh whip recordings for one
M0 → M1 → M2 chain at one target. This supersedes the earlier fresh-preliminary,
constant-model-class and 90-flight proposals. It is a protocol, not a claim
that the new collection or aircraft validation has already happened.

## Fixed scope

Use the existing 145 g UAV/17 g cable setup with unchanged controller, geometry
and tracking frame, subject to the operator confirming the actual hardware
still matches the retained model. Reference target: `[1.25, 0, 1.00]` m; nominal
tracked-origin start: `[0, 0, 1.255]` m. Use one target throughout this study.

Commands are desired PVA at 30 Hz, generated offline and exported as CSVs.
The laboratory's separate flight program executes them with onboard tracking
feedback. The repository implements no validated aircraft sender or live
launch-readiness gate. The operator uses the existing lab procedure to prepare
the vehicle and verify the actual interface, frame, onset and tracking.

Preserve the exact retained M0 model, its selected CSV, original forecast and
preliminary training/holdout roles. Do not recollect preliminary data or refit
M0. Historical M1/M2 and the 13 old whip takes remain development evidence;
they do not replace the new M1/M2 chain.

## The 20 executions

| Phase | Executions | Roles | Next step |
|---|---:|---|---|
| M0 update batch | 5 | 001/002/004 adaptation; 003/005 operational validation | Fit new M1; plan and export its command |
| M1 update batch | 5 | 001/002/004 adaptation; 003/005 operational validation | Fit new M2; plan and export its command |
| Final comparison | 10 | All final test: five M0/M2 pairs | Score physical distance and all three frozen models |

Final paired order: **M2–M0, M0–M2, M0–M2, M2–M0, M0–M2**.
Keep each pair close in time and record battery, launch and session conditions.
Use the same frozen command throughout a generation's repeats. Do not replace
a poor-distance trial to improve the result. Preserve aborted executions and
tracking failures with their reasons.

Reserve time for the final comparison. Two sequential fits plus planning and
review are the principal scheduling uncertainty. If an update cannot finish,
retain the incomplete study and report that limitation; do not rename a parent
model to make the chain appear complete.

## Recording and processing

The `deployment` branch's operator app imports the two original recordings into
the assigned take slot and automates parsing, command comparison, preparation
and reports after an explicit action. Its runbook is `docs/LAB_RUNBOOK.md`.

1. Save the original controller log and native OptiTrack CSV. Preserve their
   raw bytes, coordinates, timestamps and marker gaps.
2. Estimate timing from the two measured position streams, then inspect the
   offset, alignment error and spread. This is estimated alignment, not proof
   of a shared hardware clock or measured transport delay.
3. Review exact command identity, marker/rigid-body identity, usable free-motion
   interval, contact and intervention. Record launch/battery notes and save
   the review. The operator remains responsible for these observations.
4. Keep failed, partial or excluded observations with their reasons. Large
   target distance alone is not a reason to exclude a take.

Do not normalize away physical errors, fabricate missing markers or bridge
large tracking gaps. A physical contact changes the free-cable dynamics: keep
the task outcome, and flag/censor affected prediction or fitting intervals
under the recorded review.

## Model update and planning

Use the existing [staged full identification](../methods/FROZEN_SYSTEM_IDENTIFICATION.md):
aircraft response, aircraft residual, cable physics, cable residual, then freeze
selection before validation. Use causal initialization and the preserved raw
coordinate/observation rules. New M1 starts from retained M0; new M2 starts from
this new M1. Replay only designated training inputs. Family masses are new whip
1, prior training whip 0.5, preliminary training 0.5, normalized over present
families; take weights are equal within each family.

Operational-validation takes support review of the frozen candidate; they are
not untouched final tests. Do not tune stopping, bounds, weights or the method
to those errors and describe it as the same predeclared fit. Combined prediction
is evaluated after selection, not jointly optimized. Preserve unsuccessful
attempts; do not automatically retry until a favorable result appears.

Plan each updated model with the experiment's frozen retained-M0 planner
settings, command-seed bank and wave reference. Preserve the original source
and settings rather than rebuilding a reward from a prose description. Select
by simulation, preview the complete recovery, then export. Historical later
M2 reward changes are not defaults for this clean chain.

Retained M0 has no active cable residual; full M1/M2 enable it. This deliberately
preserves the user's baseline and tests the complete refinement procedure,
including a model-capacity change. It is not an extra-data-only comparison.

## Endpoints and analysis

For target `g` and measured tip position `p_tip`, use:

```text
d_min = minimum ||p_tip(t) - g|| over observed contiguous segments in [0, 1.5] s
```

Report distance in centimeters, closest-point time and tracking coverage. Smaller
is better. **There is no binary 5 cm paper success criterion.** Keep the existing
planner's frozen contact/ranking definition separate from this measured metric.
The common 1.5 s window can include early recovery; do not extend or shorten it
per model to obtain a more favorable value.

Show all five final M0/M2 paired distances and differences. Gaps or a truncated
recording limit the observed minimum: keep those flags visible and distinguish
complete observations from partial ones. Do not treat an unobserved interval as
a measured miss or treat a frame as an independent trial.

The supporting model endpoint is **complete command-to-tip RMS**. Evaluate
frozen M0, M1 and M2 on the same final recordings with identical command
schedules, causal history, time grid and masks over the common reviewed interval
within 0–1.5 s. Preserve exclusions and coverage. Drone error and cable error
conditional on measured attachment are separate diagnostics.

Keep original preflight forecast comparison distinct from common-history
postflight model comparison. Do not regenerate an old forecast with a later
model and call it the original. Final recordings stay outside every fit,
checkpoint choice, stopping decision, planner choice and threshold adjustment.

Report individual observations and paired mean/median differences; with only
five pairs, emphasize the measured magnitude and consistency without implying
high precision. Final M0 trials are new repetitions of a training-stage command.
Call an M2 command unseen only if its sequence actually differs from both
training-stage commands. One target and one chain do not establish generality
across independent model fits, hardware or tasks.

## Records to retain

Keep the study ledger, roles/reviews, raw files, source/model hashes, fit jobs,
stopping reasons, planner settings and time, exact CSVs, original forecasts and
all final per-take reports. The deployment workflow uses repository-relative
`experiments/`, `runs/` and `exports/` folders. Its private baseline bundle and
source-only release have different contents; record which was used.

Ubuntu/RTX 5080 is planned. Existing software checks were on Windows/RTX 4080;
fresh installation, actual aircraft timing and complete new M1/M2 fitting must
be reported from what is actually performed in the lab.
