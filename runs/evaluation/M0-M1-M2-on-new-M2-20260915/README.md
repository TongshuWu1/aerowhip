# M0 to M2: same new flights

| Prediction RMSE (cm) | M0 | M1 | M2 | M0 to M2 reduction |
|---|---:|---:|---:|---:|
| Vehicle | 8.74 | 4.36 | 4.23 | 51.6% |
| All cable markers | 10.62 | 5.08 | 5.38 | 49.3% |
| Cable tip | 12.73 | 6.65 | 6.34 | 50.2% |
| Cable tip, measured attachment | 7.90 | 4.43 | 4.08 | 48.4% |

All 11 retained M2 whips, with equal weight per whip. Three recording/battery sessions; session 2 whip 4 excluded for failure. None of these recordings trained M0, M1, or M2.

Scored interval: 0 to 1.1333333333333333 s. Shared commands, initialization history, geometry, masks and time grid. Measured OptiTrack positions are interpolated onto the shared physics grid; original raw data remain unchanged. Conditional cable scores use measured attachment motion; command-driven scores do not.

This is prediction accuracy, not target-contact performance. Three sessions are too few for a strong statistical significance claim. No fitting, model selection or command correction was performed.
