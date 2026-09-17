# Vertical M0-M4: initial-state-conditioned comparison

Exploratory post-hoc analysis. Thresholds were agreed before inspecting filtered outcome errors. No horizontal data, refitting, command changes, or manuscript edits.

## Fixed filter

- Cable shape RMS <= 3 cm across the ten tracked markers, relative to the measured attachment and the nominal hanging shape.
- Tip speed relative to the attachment <= 10 cm/s.
- Tracked vehicle position deviation from nominal hover <= 5 cm.
- Vehicle speed <= 5 cm/s. All four conditions must pass.
- Positions use the last observation strictly before command onset. Velocities use causal quadratic endpoint regression: existing 1 s cable / 0.4 s vehicle histories, 0.02 s weighting time constant.
- Shape is not projected by any candidate model. Filter membership is identical for every predictor. No outcome, target error, or hit success enters the filter.

## Retention

| Command collection | Kept / original | Recording groups retained |
|---|---:|---:|
| M0 | 8 / 11 | 3 / 3 |
| M1 | 10 / 12 | 3 / 3 |
| M2 | 6 / 11 | 3 / 3 |
| M3 | 7 / 12 | 3 / 3 |
| M4 | 10 / 12 | 3 / 3 |

## Primary: common final M4-command collection

Each row predicts the exact same flights. This collection is outside the registered training ancestry of M0-M4. It remains retrospective and conditions on near-nominal starts; it is not a new independent test.

| Predictor | Tip RMSE: all / filtered (cm) | All-marker RMSE: all / filtered (cm) |
|---|---:|---:|
| M0 | 13.86 / 13.94 | 10.78 / 10.82 |
| M1 | 6.42 / 6.49 | 4.46 / 4.50 |
| M2 | 6.14 / 6.23 | 4.71 / 4.79 |
| M3 | 6.03 / 6.15 | 4.34 / 4.40 |
| M4 | 5.67 / 5.80 | 4.38 / 4.48 |

## Secondary: all five command collections

This aggregate mixes training-exposed and unexposed recordings; it is a descriptive comparison, not a held-out ranking.

| Predictor | Tip RMSE: all / filtered (cm) | All-marker RMSE: all / filtered (cm) |
|---|---:|---:|
| M0 | 14.04 / 14.08 | 10.92 / 10.94 |
| M1 | 6.73 / 6.66 | 4.81 / 4.76 |
| M2 | 6.34 / 6.26 | 4.97 / 4.88 |
| M3 | 6.34 / 6.32 | 4.75 / 4.70 |
| M4 | 6.08 / 6.02 | 4.77 / 4.70 |

## Physical outcomes by executed command

These compare measured motion with the unchanged M0 reference, not prediction with measurement.

| Command collection | Reference RMSE: all / filtered (cm) | Fixed-time target distance: all / filtered (cm) | Minimum distance: all / filtered (cm) |
|---|---:|---:|---:|
| M0 | 16.84 / 16.54 | 31.55 / 30.51 | 12.62 / 12.69 |
| M1 | 7.92 / 7.81 | 21.94 / 21.32 | 4.87 / 4.79 |
| M2 | 7.73 / 7.26 | 13.80 / 11.46 | 4.64 / 4.47 |
| M3 | 7.86 / 6.88 | 12.65 / 11.13 | 4.72 / 3.78 |
| M4 | 7.04 / 7.09 | 13.61 / 13.60 | 3.83 / 3.47 |

All trajectory scores use [0, 1.1333333333333333) s, matching the rewritten experiment tables. The fixed strike time remains 1.1138000791100293 s. Values are equal-flight means of per-flight 3D RMSE or event distance. Sample SDs, per-recording means, and paired model differences are stored in summary.json.

No time shifting or spatial registration is applied to outcome scoring. Only the filter shape calculation subtracts attachment position to separate cable shape from vehicle translation; vehicle position is gated independently.

The existing predictors already use measured initialization. Changes after filtering indicate performance on a conditional subset, not proof that nominal initialization or downwash caused the remaining error. Small samples and session/command composition limit interpretation.

The separate 17/20 M4 contact test is untouched and is not recomputed from these target-free flights.

Artifacts: filter.json freezes inclusion/exclusion and reasons before scoring; summary.json contains full aggregates; report.json is a UI-compatible filtered M0-M4 comparison with 58-flight unfiltered baselines retained in summary.json.

UI: Evaluation -> Refresh -> this report -> All takes - development. That UI aggregate covers all retained command collections; use the common-M4 table above for the primary comparison.
