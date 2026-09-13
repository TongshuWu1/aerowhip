# Exploratory M2 subset: takes 002 and 004

This post-hoc subset was requested after reviewing the flight outcomes. Takes 001, 003 and 005 are omitted only from this sensitivity analysis. The operator suspects a hardware defect, but no independent take-specific fault evidence or outcome-independent exclusion criterion has been provided. All five takes remain in the primary report and all raw files are preserved.

| Metric | M0, all 5 | M1, all 5 | M2, all 5 | M2, selected 2 |
|---|---:|---:|---:|---:|
| Target error at original strike (cm) | 24.12 +/- 6.94 | 11.18 +/- 4.04 | 13.10 +/- 7.52 | 6.17 +/- 4.99 |
| Closest target distance (cm) | 14.73 +/- 6.02 | 7.83 +/- 1.82 | 8.60 +/- 6.55 | 4.41 +/- 2.51 |
| Tip-reference RMSE (cm) | 18.09 +/- 1.80 | 16.06 +/- 2.34 | 13.28 +/- 2.60 | 10.57 +/- 0.78 |
| Quadrotor-reference RMSE (cm) | 15.66 +/- 3.58 | 11.17 +/- 2.87 | 12.63 +/- 5.65 | 9.00 +/- 0.94 |
| Tip speed at original strike (m/s) | 5.98 +/- 0.53 | 5.85 +/- 0.32 | 5.82 +/- 0.79 | 5.84 +/- 0.71 |

Values are equal-take means +/- sample standard deviation. Intervals, original reference, clock alignment and velocity estimation are unchanged from the primary analysis.

The selected pair has lower mean target and trajectory errors than the five M1 flights. This describes these two executions; it does not establish improved typical performance or repeatability. Selection after observing outcomes can bias the comparison, and two repetitions do not characterize the full M2 variability.

Take 002 retains the existing clock-alignment limitation: the two recording halves disagree by 9.46 ms. Its fixed-time target distance changes from 2.64 cm to 6.80--7.15 cm under +/-10 ms shifts. No timing shift was chosen to improve the subset results.

[Primary five-take report](../REPORT.md).
