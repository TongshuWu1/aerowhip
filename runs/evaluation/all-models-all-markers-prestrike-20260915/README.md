# All 10 cable markers: prediction RMSE (cm)

| Predictor | M0 flights (11) | M1 flights (12) | M2 flights (11) | M3 flights (12) | All 46 |
|---|---:|---:|---:|---:|---:|
| M0 | 11.81 | 10.32 | 10.25 | 10.08 | 10.60 |
| M1 | 5.18 † | 4.13 | 4.83 | 4.30 | 4.59 |
| M2 | 5.63 † | 3.91 † | 5.18 | 4.56 | 4.80 |
| M3 | 5.51 † | 4.20 † | 4.72 † | 4.32 | 4.67 |
| M4 | 5.43 † | 4.05 † | 4.85 † | 4.22 † | 4.62 |

† Recordings in the model's training ancestry; retention diagnostics. Other cells are recordings not in that model's training ancestry. Compare predictors within a column: commands differ between flight collections. The all-46 column mixes training and unseen data.

Score window: whip onset to the fixed planned strike at 1.113800079 s; physics-grid samples before strike only. No post-strike, braking, recovery or hover enters the score. Models receive recorded commands and measured causal initial state, then predict the vehicle and all 10 tracked cable markers. The attachment is excluded, and the tip has no extra weight. Each whip has equal weight in the table. Missing observations stay masked identically for every model. No physical target was present.

## Each marker across all 46 whips (cm; mixed training and unseen data)

| Predictor | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 9.20 | 9.51 | 9.62 | 9.69 | 9.76 | 9.98 | 10.34 | 11.22 | 12.25 | 13.49 |
| M1 | 3.99 | 4.00 | 3.97 | 3.95 | 4.03 | 4.21 | 4.48 | 4.93 | 5.46 | 6.09 |
| M2 | 4.06 | 4.15 | 4.24 | 4.33 | 4.51 | 4.68 | 4.88 | 5.16 | 5.47 | 5.84 |
| M3 | 3.86 | 3.94 | 3.99 | 4.04 | 4.21 | 4.41 | 4.68 | 5.10 | 5.55 | 6.01 |
| M4 | 4.17 | 4.17 | 4.12 | 4.11 | 4.22 | 4.37 | 4.57 | 4.90 | 5.26 | 5.65 |

c10 is the tip. Combined RMSE is computed from squared position errors, not by averaging the ten marker RMSE values. Per-take, per-session and measured-attachment cable diagnostics are saved in report.json and summary.json. These are descriptive results, not a statistical significance claim. M4 training uses 75/25 weighting; evaluation uses equal sample weights for all 10 markers for every model.

UI: refresh Evaluation, choose this report, select All takes · development and the command-driven all-marker metric.
