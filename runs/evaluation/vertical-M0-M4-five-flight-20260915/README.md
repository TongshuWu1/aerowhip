# Five-flight physical evaluation: vertical M0-M4

Retrospective subset: five filtered physical flights per executed command, balanced 2/2/1 across its three recording groups. Selection uses fixed seeds 20260915+i for Mi and was saved before calculating these subset results. It uses no prediction, tracking, targeting, or contact outcome. Earlier full-cohort outcomes were already known; this is not a preregistered prospective study.

| Command | n | Tip-reference RMSE (cm) | Fixed-time distance (cm) | Minimum distance (cm) |
|---|---:|---:|---:|---:|
| M0 | 5 | 17.32 +/- 2.24 | 31.53 +/- 3.49 | 12.45 +/- 1.60 |
| M1 | 5 | 7.87 +/- 0.75 | 22.62 +/- 4.74 | 4.89 +/- 0.74 |
| M2 | 5 | 6.58 +/- 0.90 | 11.57 +/- 7.67 | 3.80 +/- 1.79 |
| M3 | 5 | 6.96 +/- 0.60 | 10.78 +/- 6.33 | 3.80 +/- 1.51 |
| M4 | 5 | 7.31 +/- 0.73 | 13.16 +/- 7.81 | 3.46 +/- 2.66 |

Means and sample standard deviations are across flights, not frames. Recording-group means and exact IDs are saved. Physical metrics were recomputed from immutable measured traces with the existing time window, reference, masks, and target-event definitions; all per-flight values matched the earlier evaluation within 1e-12 m.

The common ten-M4-flight tip prediction evaluation is unchanged and was independently rechecked for every predictor. No model fitting, command changes, horizontal analysis, or contact-test filtering was performed. Full filtered and unfiltered physical baselines remain in summary.json.
