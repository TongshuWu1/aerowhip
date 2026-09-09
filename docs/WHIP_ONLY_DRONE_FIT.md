# Initial whip-only drone response model

**Subsequent complete-maneuver validation found large errors.** See `data/historical_model_runs/20260908-014458-686535-combined-whip-heldout/REPORT.md` and its 3D/error plots. Both-residual tip RMSE is 47.7–72.3 cm over uninterrupted 1.5 s maneuver predictions, and about21cm within the logged whip itself. The sampled one-second window averages below do not establish complete-whip accuracy. That validation refits drone folds with cable geometry that also excludes the held-out whip.

The user selected a narrower domain for the initial drone model: recordings from virtual force-policy execution in simulation, exported as P/V/A and executed through cmdFullState. Only `whip1_001`, `whip1_002` and `whip1_003` enter the nominal drone response or drone NN losses. Preliminary trajectories remain eligible for cable physics and cable residual fitting.

Run: `data/historical_model_runs/20260908-013552-736857-whip-only-drone`.

Both the cf7-origin and cable-attachment response models were fitted from scratch, including nominal gains, effective delay and a zero-initialized small NN. Each of three development fits excludes one entire whip. Final models use all three whips. The inputs, source hashes, masks, window lists, saved weights and evaluations are preserved in the run folder; original measurements and the previous pooled models were not changed.

The attachment offset is frozen from the separately fitted cable geometry so the models agree about the attachment location. This geometry is shared development calibration, not an independent held-out estimate. No preliminary command/response samples enter drone fitting. New prepared historical protocols explicitly specify `drone_response_takes`; tests verify that preliminary takes cannot enter that selection.

## One-second prediction checks

Euclidean position RMSE, with each whip excluded from its own response fit:

| Attachment trajectory | Nominal response | With drone NN |
|---|---:|---:|
| whip1_001 | 2.905 cm | 3.336 cm |
| whip1_002 | 14.544 cm | 11.928 cm |
| whip1_003 | 1.768 cm | 2.450 cm |
| Equal-take mean | 6.406 cm | 5.904 cm |

The mean improves because the difficult second take improves; the other two worsen. Treat this as a preliminary model with limited evidence. The cf7-origin model's corresponding mean is 5.502→5.144 cm. These errors are not target-hitting errors, and these changed fitting splits do not establish superiority over the old pooled fit.

The final attachment model selects an effective delay of 60 ms. Vertical velocity gain and Y/Z feedforward gains reach their configured bounds. These are empirical prediction parameters, not firmware gains to copy to the vehicle. The delay also includes timestamp/alignment assumptions.

The historical whip stream switches to hold near 0.67 s; it does not supply three fully executed 0.8 s maneuvers. Logged hold is retained as actual input, and missing intended commands are not invented. With only three maneuvers, the model's coverage is narrow even though each recording contains many hover samples.

## Saved candidate and verification

`candidate_bundle/model.json` combines the unchanged cable candidate with the new whip-only attachment response/NN. It is saved separately and is **not active**. Existing combined accuracy reviews used the old pooled drone model and are not applicable to this replacement. Cable regression from the earlier review also remains unresolved.

Fourteen targeted scope, drone-model and historical-data tests passed. The replacement full model passed a 1,024-case numerical probe on Windows/RTX 4080 in about 19.2 seconds, with zero numerical failures and no optimizer updates. Numerical execution is not a flight or accuracy validation. Current calibration, selected PPO, controller and logger remain unchanged; no PPO training or flight was started.
