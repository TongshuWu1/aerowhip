# Historical starting model and prospective adaptation

Decision recorded 8 September 2026. The user clarified the purpose of all
current recordings: use them carefully to establish one preliminary model,
then retire outdated material from active work and collect new recordings
through the corrected pipeline for subsequent adaptation. This document
records that workflow; it does not launch a fit, PPO training or flight.

Latest inventory and next-session procedure:
[Data lifecycle and recording guide](DATA_LIFECYCLE_AND_RECORDING_GUIDE.md).
The eight preliminary takes remain enabled in the legacy manifest. A separate
nominal drone fit now consumes explicit review/quality/phase masks on three
whips; the [attitude correction](ATTITUDE_MAPPING_CORRECTION_20260908.md) resolves
the observed attitude-domain failure, with post-hold limitations still explicit. The old loaders/UI are not globally
upgraded by this fitter. Retirement of legacy samples after M0 is saved
remains a required implementation step, not an already enforced selection rule.

## Preliminary model M0

1. Establish the model definition before fitting, as previously requested. Keep
   tracked drone origin, rotating cable attachment and flexible first cable
   span distinct. Specify how the execution model predicts future orientation
   and attachment motion; future measured orientation cannot be an input to
   an open-loop prediction. The standalone nominal pose engine now implements
   this response and has passed implementation review. The first nominal fit
   and subsequent attitude correction are assessed. Residual identification,
   cable fitting and combined-model assessment remain, with the separate
   post-hold limitation explicitly retained.
2. Use valid historical preliminary and whip intervals for cable physics and
   cable residual fitting, including fig8vertical_002 under the latest scope.
   Use measured attachment motion as the cable boundary to avoid attributing
   drone tracking error to cable parameters. Preserve measured masses and
   ruler dimensions; record exclusions as masks without removing raw samples.
3. Use only whip1_001/002/003 for the nominal drone response and small drone
   residual, as requested. Input is the P/V/A actually sent through FullState,
   and the measured response is at the tracked drone origin. This estimates
   effective closed-loop command tracking; it does not identify motor/thrust
   physics from a measured force signal.
4. Consume the phase-aware processed version
   `data/adaptation_rounds/adaptation0/processed/20260908-041031-260837` explicitly.
   Preserve surrounding holds as their own regimes; include complete maneuver
   transients in fitting and reporting. Replace the historical sampler that
   mostly selected hover. Take 3's different post-hold is intentional. Match
   only the 20 observed maneuver CSV rows, never the unexecuted 25-row tail.
5. Fit nominal terms first, then regularized residuals; check complete recursive
   trajectory predictions against each take. Separate cable replay with
   measured attachment from combined drone-plus-cable prediction. Report
   maneuver and hold errors separately. Any historical leave-one-take-out
   checks are development checks because these recordings already informed
   the method. Do not present them as new independent experimental evidence.
6. Save the chosen preliminary bundle as M0 with geometry, parameters, residual
   weights, preprocessing version, masks and fit diagnostics. Preserve failed
   candidates and record the limitations of the selected bundle.

The existing historical fitting loader still pins the previous processed data,
and its attachment-response fit uses a fixed initial command translation.
Re-running that entry point unchanged is not the corrected M0 procedure above.
The present processing revision did not resolve those model/fitting issues.

## New policy and recordings

After the model definition and M0 fit, implement the agreed new 30 Hz force
policy and 30 Hz FullState command pipeline, then train a fresh PPO. Do not
retime the old 20 Hz checkpoint or automatically resume stopped training.

The policy generates a frozen force sequence. The virtual physical simulation
turns that sequence into the desired P/V/A command trajectory. The execution
model predicts how the real drone and attached cable respond to those commands.
The desired command trajectory is the exported flight input; the predicted
actual response is not substituted for the command reference. Preserve this
distinction in saved files and comparisons.

Freeze M0, the new policy and each exported command sequence before collection.
Keep full takeoff/hold/maneuver/recovery recording coverage where available and
label the actual executed phases. Record the full command stream and measured
tracking/cable motion without inventing unlogged commands. Collection remains
an experiment, not proof that M0 is correct.

## Adaptation after the first new collection

First evaluate M0's frozen prediction against the new execution, including
tracking and cable-tip errors during the maneuver. Then use the new data to fit
M1. This recording is prospective evidence for M0 before it is used for fitting;
after fitting M1 it is training/development data for M1. A subsequent fresh
execution assesses M1 and the resulting policy.

Continue the same record → evaluate frozen prediction → adapt → retrain or
update policy → execute cycle with explicit model and policy versions. The
active adaptation dataset should contain recordings from the corrected route.
Any later reuse of a legacy interval must be explicit rather than silently
mixing old and new settings.

## Retiring old data

After M0 is fitted and saved, retire legacy inputs and superseded outputs from
the active adaptation workflow. Retain one immutable provenance archive tying
M0 to the raw logs, masks, processed inputs, source versions and fit results.
Retirement means exclusion from default active fitting and UI selection; it
does not require permanent deletion of the only reproducible copy. Preserve
the existing historical study snapshots, selected policy and active calibration
until a replacement is explicitly selected. No archival move or deletion has
been performed by this planning clarification.
