# New M7: completed horizontal refit

- **M6:** the former M7 predictor and its unchanged command, forecast, and eight measured flights. The original intermediate M6 is archived; its contribution to model ancestry is preserved.
- **New M7:** initialized from this renamed M6 and fitted on its eight latest whips (two recordings), with preliminary training replay. Earlier whipping batches were not replayed.
- **Convergence:** six consecutive checks without more than 1% improvement relative to the last meaningful-progress anchor. Neural checks occur every five updates, after the existing minimum of 40 updates. The absolute best checkpoint is retained even if its improvement is below 1%.
- **Vehicle residual:** stopped at update 65; selected update 65; loss 3.238935 → 3.132980.
- **Cable residual:** stopped at update 75; selected update 70; loss 0.855116 → 0.812612.
- Both residual states reached six stale checks and passed numerical gradient verification. Physical parameter stages and full prediction checks completed.

## Prediction checks

Equal-flight means on the eight training whips, using the same measured initialization and recorded commands:

| Metric | M6 | New M7 |
|---|---:|---:|
| Vehicle prediction RMSE | 6.68 cm | 5.62 cm |
| Command-driven tip prediction RMSE | 14.95 cm | 12.62 cm |
| Tip RMSE with measured attachment | 3.26 cm | 3.01 cm |

These are **training diagnostics**, not new flight performance or held-out whipping results. They are distinct from tracking the desired trajectory. Preliminary holdout vehicle RMSE changed from 9.63 to 9.79 cm; conditional tip RMSE changed from 5.12 to 4.83 cm.

## Verification and naming

An unrelated live paper-figure script changed after the fit began. The optimizer finished normally; the live-source guard then stopped final evaluation. Final prediction and preliminary holdout checks were completed using the exact saved `source_snapshot`, without changing the weights or numerical criteria. See `verification_resume.json` and `frozen_verification.log`.

The new M7 is registered in the model catalog. The actual historical update count remains in `generation_index`; public model IDs follow the requested M6/M7 naming. The archive at `archive/horizontal-renumber-20260916` preserves original files, the omitted intermediate model, and the migration manifest.

The M6 command CSV, physical recordings, and frozen preflight forecast retain their original numeric bytes. The UI was checked for the two model labels, eight M6 flight replays, command-correction view, rehearsal scrubbing, and the fitting criterion. Refresh the application lists to see the updated names.
