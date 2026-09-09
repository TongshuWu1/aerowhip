# Nominal drone fitting assessment

Status: **NOMINAL_FIT_ASSESSED_MODEL_LIMITATIONS**. The nominal-only identification
and all three development comparisons are complete; this is not a complete M0
bundle or a deployable model.

Read the [full result report](../../../docs/NOMINAL_DRONE_FIT_RESULTS_20260908.md)
and [fitting protocol explanation](../../../docs/NOMINAL_DRONE_FITTING.md).

Machine-readable evidence: `protocol.json`, `mask_report.json`, `results.json`,
`verification.json`, `test_verification.json`, `acceleration_diagnostics.json`.
Each fit/fold contains traces, parameters or a rejected-candidate record,
assessment and prediction arrays. `nominal_fit_review.png` shows the trajectories;
an absent orientation curve means unavailable prediction, not zero error.

Selected GUI/model/calibration/PPO files remain unchanged. No drone NN, cable
fit, PPO/SAC training, retirement of old measurements or flight was performed.
