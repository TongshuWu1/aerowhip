# Combined held-out whip validation

**Outcome: the current combined model does not predict the complete whip accurately enough to treat it as a validated training model.**

Run: `20260908-014458-686535-combined-whip-heldout`. All three leave-one-whip-out predictions are complete. No active calibration, PPO, controller or logger was changed.

## Protocol

Each whip is excluded from both component fits and cable geometry. Existing matching cable folds were reused; the drone response/NN were refitted on the other two whips with that fold’s geometry. These are held-out parameter fits, but still development checks: the recordings previously informed methodology and are not independent paper evidence.

Prediction begins at the first logged nonzero velocity/acceleration command and runs continuously for 1.5 s. The initial drone and cable state uses only past measurements. Thereafter the inputs are logged cmdFullState P/V/A, including the actual switch to hold near 0.67 s. No measured state is injected during the prediction. The separate measured-attachment diagnostic intentionally uses future measured attachment positions to isolate cable error; it is not an open-loop validation result.

All three recordings support this common horizon. Cable-quality masks exclude 9, 0 and 1 frames respectively without resetting the prediction or filling measurements. Commands and drone inputs are valid throughout the tested interval. A second run assumes a straight downward cable moving initially with the measured drone velocity, matching the planned drone-only initialization assumption.

## Both residuals: per-take results

| Take | Attachment RMSE, 1.5 s | Tip RMSE, 1.5 s | Tip RMSE during 0–0.67 s | Tip position error at 0.78 s |
|---|---:|---:|---:|---:|
| whip1_001 | 26.58 cm | 47.73 cm | 21.15 cm | 65.39 cm |
| whip1_002 | 23.52 cm | 72.30 cm | 21.98 cm | 63.68 cm |
| whip1_003 | 26.39 cm | 61.39 cm | 20.63 cm | 59.72 cm |

## Component comparisons

| Model combination | Mean tip RMSE over 1.5 s |
|---|---:|
| physics only | 67.67 cm |
| cable nn only | 60.43 cm |
| drone nn only | 67.61 cm |
| both residuals | 60.47 cm |
| measured attachment diagnostic | 22.75 cm |
| Both residuals, assumed straight initial cable | 60.78 cm |

The cable NN helps the average relative to the new physics-only chain, but substantial error remains. Adding the drone NN does not consistently improve this maneuver-wide prediction. Supplying the real attachment trajectory reduces error substantially, identifying drone-response error as a major contributor, while the remaining cable error shows that cable dynamics also need work. Similar measured-initial versus straight-initial results mean obtaining the initial cable state alone would not solve the mismatch in these trials.

The earlier few-centimeter averages used sampled windows with measured initialization at each start, many outside the complete maneuver. They understate the error of uninterrupted prediction from maneuver onset. This test does not tune models after examining these results.

## Intended hit time and target

Target distances below use the historical intended target [1, 0, 1.4] m, not an independently surveyed physical object. The 0.78 s intended hit time occurs after the recorded switch to hold; these are not validations of an unexecuted 0.8 s command sequence. Clock alignment is approximate and includes logging latency.

| Take | Predicted distance to target at 0.78 s | Observed distance to target at 0.78 s | Closest observed distance over 1.5 s |
|---|---:|---:|---:|
| whip1_001 | 15.90 cm | 79.18 cm | 34.02 cm at 0.98 s |
| whip1_002 | 97.02 cm | 73.74 cm | 33.57 cm at 0.97 s |
| whip1_003 | 22.12 cm | 80.64 cm | 29.63 cm at 1.00 s |

## Next step

Verify command timing and fit drone transient response on continuous whip-centered rollouts, then address cable dynamics using measured attachment inputs and repeat the held-out checks. Do not train a new PPO assuming the present complete model is accurate. Preserve these negative results and all raw measurements; do not discard a take because it has large error.

Five targeted split/masking/command-history tests passed. Numerical validation ran on Windows/RTX 4080. Original source and snapshot checksums are preserved. No real flight was tested.

Artifacts: `validation/*/trajectories.npz`, `validation/*/metrics.json`, `validation_hanging/`, `summary.json`, and `validation_overview.png`.
