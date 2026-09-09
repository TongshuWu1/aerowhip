# Historical-data baseline: checked proposal

This proposal has now been executed. See [completed fit and mixed results](HISTORICAL_MODEL_FIT_RESULTS.md) and [reproduction workflow](HISTORICAL_MODEL_FITTING.md). The complete candidate is saved but was not applied because the previous-calibration comparison does not establish improved cable-tip prediction, particularly on historical whips.

7 September 2026 local. The user now requests reuse of all preliminary and whip recordings, including the previously protected fig8vertical_002, to build the starting model for a new policy and subsequent real-data improvement. This authorizes this inspection despite the historical protection. No roles, active models, checkpoints or training jobs were changed during this assessment.

User subsequently confirmed the preliminary and whip recordings use the same drone and settings. Treat them as the same reported setup for the proposed pooled fit; retain source-session metadata and check measured frame/geometry consistency rather than inventing separate controller configurations.

## Data checked

Read the eight processed preliminary take.npz files and the three whip dataset.npz files in adaptation0/processed/20260907-194802-922485. Preliminary duration totals 326.85 s; whip recording duration totals 44.49 s (most is not the active whip). Existing masks mark 99.32–100% of preliminary frames and 99.73–99.80% of whip frames as having complete valid cable/drone data. These masks are not independent verification of marker identity or contact-free motion.

Median intermarker chord lengths are similar across groups: first c1–c2 interval approximately 84–85 mm and subsequent intervals approximately 100 mm. This supports geometric compatibility but does not establish unchanged cable material, masses, marker assignments, attachment or controller settings. Measured chord length is not automatically cable arc length.

Preliminary recordings contain logged full-state P/V/A commands, with command-valid coverage about 86–98%. The preserved source logger confirms these are FullState messages, not measured thrust. Thus usable preliminary command/motion intervals could also support drone tracking identification, conditional on frame, time and controller compatibility.

All three whip quality reports show 20 distinct maneuver command samples matching the intended prefix, followed by hold around 0.67 s; the intended CSV extends to 0.8 s. Fit the actual logged commands, including that transition. Do not fabricate execution of the missing tail. Whip clock alignment remains approximate; contact/intervention and body/attachment mapping require review. These concerns affect different training objectives differently: cable fitting can use synchronized OptiTrack attachment and cable motion directly without relying on command-clock alignment.

## Proposed fit

1. Keep original logs and historical splits immutable. Create a new development protocol admitting every historical take, with explicit session/controller metadata and usable intervals. Missing data and interventions remain excluded from the relevant losses; failures themselves are retained.
2. Fit identifiable cable physical quantities using measured attachment motion as the boundary and measured cable markers as targets. Preserve measured masses and ruler geometry unless evidence warrants revision. Profile stiffness/internal-damping sensitivity; do not free every physical parameter just because a NN is available.
3. Fit the bounded cable NN with the physical stage frozen. The requested external-drag convention is NN-only: physical external drag is zero throughout the new physical/residual candidate, and there is no 0.3/s initialization. Compare the combined result with the preserved historical baseline. Balance takes and motion regimes so lengthy hovering does not overwhelm whipping.
4. Fit nominal effective full-state tracking response and then a small drone NN using logged commands and observed drone trajectories. Use preliminary sessions in the final shared model only if compatible; otherwise retain their value through separate session models or pretraining followed by current-setup fitting. This is not identification of motor/thrust physics from nonexistent force measurements.
5. Diagnose prediction with entire takes held out in rotation, then freeze settings and refit the final starting bundle on all compatible historical takes. Rotated checks are development diagnostics. After this final fit, none of these historical records is an independent test set. Freeze the bundle and policy before collecting the next flights for prospective assessment.

## Required PPO integration

The current PPO environment uses the force-driven point-mass/cable simulator. The cable NN can run there, but the drone tracking residual is standalone. Merely fitting both networks does not produce the requested complete training environment.

Use two stages: PPO force sequence -> virtual force/cable rollout -> exact 30 Hz exported reference -> effective drone tracking model + drone NN -> predicted attachment trajectory -> cable physics + cable NN -> predicted strike reward. Export the commanded reference from the first stage, not the predicted tracking-error trajectory. Maintain frozen open-loop policy semantics: predictions may be computed offline; measured execution feedback must not enter the actor during the maneuver.

The effective drone model already absorbs the fixed cable load present in its training data. In this first empirical execution model, prescribe its predicted attachment path to the cable; do not add cable reaction to that same drone predictor a second time. This is a one-way effective surrogate for the recorded setup, not a separately identified, fully coupled actuator model. Predicting attachment motion from cf_7 origin also needs a consistent orientation/offset treatment.

The current standard physical-fit path searches scalar external drag while freezing EI/Cb, and the cable NN fitter excludes whip data and requires fixed preliminary splits. Those paths need changes for this proposed protocol. Previous rejected cable candidates must not be relabeled as passing simply to enable every component.

## Assessment

Recommended as an initial development baseline, subject to the compatibility checks and integration above. Historical failures are useful motion observations. They cannot by themselves demonstrate improvement of a new policy; fresh executions provide that assessment. No fitting or PPO training was launched by this check.

Related methodological precedent: Gao et al., [Sim-to-Real of Soft Robots with Learned Residual Physics](https://arxiv.org/abs/2402.01086), combines physical simulation with learned residuals from sparse marker measurements. It supports the general approach, not validation of this drone/cable architecture.
