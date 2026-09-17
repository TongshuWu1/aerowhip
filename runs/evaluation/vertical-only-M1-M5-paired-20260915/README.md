# Original vertical whip: paired M1–M5 evaluation

58 repetitions, 15 recording groups, five executed command versions (M0–M4). No horizontal/curved-side recordings or M6 model are included.

## Command-driven tip prediction RMSE (cm)

| Predictor | M0 | M1 | M2 | M3 | M4 |
|---|---:|---:|---:|---:|---:|
| M1 | 6.66 † | 5.92 | 6.12 | 5.72 | 5.85 |
| M2 | 6.51 † | 5.37 † | 5.91 | 5.62 | 5.71 |
| M3 | 6.73 † | 6.02 † | 5.57 † | 5.76 | 5.75 |
| M4 | 6.39 † | 5.65 † | 5.37 † | 5.21 † | 5.29 |
| M5 | 6.46 † | 5.54 † | 5.40 † | 5.35 † | 5.07 † |

† The recordings occur in the model’s registered training ancestry: these cells are training/retention diagnostics. Other cells have no registered training-source overlap; this retrospective audit does not make them a pristine final test. Compare models within columns. Commands differ between columns; diagonal trends cannot isolate model accuracy.

## Evaluation contract

- Each model predicts the same recorded commands using the same measured physical initial state and causal pre-command initialization procedure. Model-specific hover compensation uses that same pre-command history.
- Primary scores preserve the historical pre-strike window [0, 1.1138000791100293) s; no model-specific temporal shifting or spatial registration.
- Tip and all-ten-marker 3D RMSE are separate. All-marker scoring weights each valid marker-time observation equally, with no extra tip weight.
- Conditional cable scores use measured attachment motion; these are component diagnostics, not full command-to-motion prediction.
- Missing observations use identical masks for every model. Per-whip values and per-recording-group means are retained; no frame-level confidence intervals or significance claims.
- Event diagnostics use the shared reviewed command interval [0, 1.1333333333333333) s. Closest approach is found on adjacent valid piecewise-linear segments only; endpoint minima are flagged as censored.
- Fixed planned-time prediction error is separate from closest-approach timing/position error. Event matching does not shift trajectory RMSE.
- The target was virtual, not physically present. Distances are not contact or hit-rate evidence. Velocity-based event scores are omitted rather than introducing an unreviewed derivative filter.
- All models and raw/prepared inputs are frozen and hash-verified. Current inference code is recorded. This is new retrospective inference, not a recovered original preflight forecast.

## Paired comparisons on the next collection (no registered training overlap for either model)

| Earlier → updated model | Collection | Tip RMSE before → after (cm) | Paired mean change (cm) |
|---|---|---:|---:|
| M1 → M2 | M2 | 6.12 → 5.91 | -0.20 |
| M2 → M3 | M3 | 5.62 → 5.76 | +0.14 |
| M3 → M4 | M4 | 5.75 → 5.29 | -0.46 |

**M5 has no unseen original-vertical-whip collection here.** It was trained on the latest M4 flights and inherits earlier training ancestry. A new original-whip collection is needed to independently assess M4→M5. Existing data cannot manufacture this missing comparison.

## Measured task outcomes (virtual target; not physical hit rates)

| Command collection | Planned-time target distance (cm) | Minimum target distance (cm) | Endpoint minima |
|---|---:|---:|---:|
| M0 | 31.55 | 12.62 | 0/11 |
| M1 | 21.94 | 4.87 | 0/12 |
| M2 | 13.80 | 4.64 | 0/11 |
| M3 | 12.65 | 4.72 | 0/12 |
| M4 | 13.61 | 3.83 | 0/12 |

These are measured outcomes of different commands, stored independently of which model is being scored. Collection order and session effects are not randomized retrospectively. They do not isolate the effect of model refinement.

Artifacts: `report.json` (per-whip model errors and events), `summary.json` (collection/session means and paired differences), `measured_task_outcomes.json` (measured virtual-target outcomes, stored once per flight), `evaluation.png` (tip and all-marker matrices).

UI: Evaluation → refresh → this report → All takes · development. Inspect the training-exposure labels in this report.
