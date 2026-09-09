# adp1 model and adaptation isolation audit

The evidence points primarily to a mismatch in the drone/controller response, but does not prove a power-system fault or eliminate all model-design limitations. No parameters were fitted, no policy was changed, no CSV was exported, and the original preflight ghosts remain untouched.

## Same commands and measurements: isolate the boundary motion

All values below are mean per-flight 3D RMS during0–1s, four cf3/adp1 takes001–004.005 excluded. Both models use the same normalized measured pre-hover initialization and exact logged FullState command receipts. These diagnostic replays are distinct from the original saved forecast used in the real-flight comparison.

| Diagnostic | M0 | M1 |
|---|---:|---:|
| Drone prediction RMS |27.02cm|21.98cm|
| Coupled drone+cable tip prediction RMS |18.52cm|21.77cm|
| Cable-tip RMS with measured rotated attachment supplied |7.11cm|6.66cm|
| All cable markers RMS with measured attachment |3.95cm|3.82cm|

Supplying measured drone motion reduces M1 tip error by about69%. This uses future measured attachment as a diagnostic input; it is not an open-loop prediction result. M1 improves drone RMS and measured-boundary cable RMS over M0 on these identical commands, yet coupled tip RMS worsens. Cable motion is sensitive to the direction and timing of root errors; component RMS improvements do not guarantee a better coupled tip path. Do not hide this result or infer that adaptation is globally correct.

## What was checked

- Published M1, PPO frozen model, and flown rehearsal have identical drone parameters, drone residual tensors, cable physics/geometry and cable residual checksum.
- Seven principal dynamics/reference/rollout modules are byte-identical to the PPO source snapshot.
- Exact CSV matches rehearsal20260909-030657-671710 and checkpoint424b5be3c20b8ba1b7d067eb4f53ee1a10045a860ff832d7853c02b565dd7fbb.
-39 focused tests passed across geometry, drone response, native30Hz training/reference handling and research CSV export. Timing/normalization had15 tests pass previously,3 historical tests skipped. Tests do not verify the physical cf3 mounting or current firmware configuration.
- State uses only OptiTrack; controller XYZ is used only for the explicitly authorized time matching. Per-take Z shifts are applied once to drone and cable. Missingness preserved.
-±10ms timing sensitivity changes saved-forecast drone RMS by only a few millimetres, leaving the discrepancy near20cm. This is a sensitivity range, not a certified clock-error bound.
- All protected inputs remained unchanged; see preservation_check.json. Windows local Python and CUDA runtime tested; no remote aircraft/Ubuntu validation.

## What the drone actually does differently

Against the saved M1 ghost, per-axis RMS is about20.6cm X,2.9cm Y,6.1cm Z. Mean measured-minus-predicted X is approximately-17.6cm, and Z approximately-4.9cm across the whip. Thus the drone lags the predicted forward travel and remains lower even after constant hover normalization. Reduced available thrust is consistent with this pattern, but controller gains, thrust mapping, delay and model generalization are also possible explanations.

## Model-design and adaptation limitations remain

1. The drone model is an effective loaded-drone PVA response model with PD/feedforward gains, delay, attitude response and bounded acceleration NN. It has no battery voltage, motor PWM, thrust saturation or changing controller-integral state. Translational acceleration is not constrained to the estimated thrust axis. It cannot identify a power defect from position logs.
2. The cable is driven by the predicted rotated attachment. Loaded-drone response already contains cable-load effects empirically; the code does not inject another explicit cable reaction into the fitted drone model. This avoids straightforward double counting but is a local approximation when the maneuver/load pattern changes.
3. Hover compensation is estimated from nominal dynamics, while prediction adds the NN. M1 NN at the measured hover contributes about[-0.375,-0.005,+0.140]m/s² on average. An offline sensitivity that offsets compensation by the mean hover NN does NOT fix adp1: drone RMS rises from21.98cm to about25.11cm. Removing the drone NN similarly worsens RMS to about25.21cm. This inconsistency deserves joint initializer/fit design review, but is not a demonstrated fix and was not patched into the frozen model.
4. M1 residual fitting stopped at80 drone/24 cable updates with still-decreasing training losses. Nominal kp_xy and kd_z reach their search bounds, and cable Cb is at its grid's lower edge. Fitting is a local bounded baseline, not established convergence or global identification. Future stopping changes are documented separately; no new fit was run.
5. cf7→cf3 changed the vehicle; physical tracking-origin/attachment geometry and controller/firmware equivalence are not verified by a CSV. Initial cable projection corrections were5.6–8.3mm. Hover normalization uses future post-hold data and does not remove thrust/phase drift.

## Recommended next experiment

Repeat the exact frozen M1 CSV on cf3 after addressing its hover/power behavior, preserving the same controller configuration and logging the actual available PWM/voltage if possible. Compare against this frozen prediction before adapting again. Until then, do not declare the problem exclusively hardware or retrain the cable residual to absorb drone tracking error.

## Artifacts

results.json and M0_*/M1_*.npz contain measured-boundary and coupled predictions; drone_sensitivity.json contains the initializer/NN sensitivity; contract_and_timing_checks.json contains identity and timing checks. No five-fold fitting, full recovery study, or heavy gradient experiment was performed.
