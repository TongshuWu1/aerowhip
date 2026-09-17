# All ten cable markers on unseen M4 flights

| Predictor | RMSE (cm) |
|---|---:|
| M0 | 10.461 |
| M1 | 4.185 |
| M2 | 4.506 |
| M3 | 4.178 |
| M4 | 4.165 |

All models predict the same 12 whips in three battery recordings. None of these recordings trained M0-M4. Scores cover onset to the fixed planned strike at 1.1138000791100293 s, excluding post-strike motion. Each valid marker-time sample has equal weight within a whip; per-whip RMSE is then averaged equally. c10 is the tip and gets no extra weight.

| Predictor | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 9.008 | 9.341 | 9.502 | 9.553 | 9.567 | 9.786 | 10.132 | 11.173 | 12.278 | 13.345 |
| M1 | 3.382 | 3.445 | 3.418 | 3.508 | 3.483 | 3.784 | 4.056 | 4.643 | 5.341 | 5.847 |
| M2 | 3.475 | 3.628 | 3.739 | 3.981 | 4.059 | 4.433 | 4.657 | 5.090 | 5.551 | 5.714 |
| M3 | 3.244 | 3.363 | 3.370 | 3.498 | 3.500 | 3.818 | 4.091 | 4.752 | 5.417 | 5.752 |
| M4 | 3.628 | 3.672 | 3.623 | 3.672 | 3.620 | 3.878 | 4.081 | 4.586 | 5.078 | 5.293 |

Session 3 has TF logger gaps; the complete command-event log supplies command timing and native 100 Hz OptiTrack supplies measured positions. Clock alignment is estimated. This is prediction accuracy, not target-contact performance. Three battery recordings do not establish statistical significance. No M5 fitting was performed.
