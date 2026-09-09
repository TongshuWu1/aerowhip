# Geometry audit — existing recordings only

No fit, optimization, data rejection or training. All measurements and configurations were preserved.
Statistics cover each entire recording, not only its maneuver. A 2 mm excess is a diagnostic threshold, not a newly applied mask.

| Take | Rigid marker spread using R^T (mm) | Using R instead (mm) | Candidate first-span median (mm) | Candidate excess >2 mm / valid |
|---|---:|---:|---:|---:|
| fig8_001 | 1.727 | 5.237 | 59.58 | 0/4285 |
| fig8_002 | 2.410 | 9.997 | 60.03 | 11/3035 |
| fig8_003 | 4.663 | 20.185 | 59.23 | 502/5729 |
| fig8vertical_001 | 2.398 | 7.593 | 58.25 | 0/6976 |
| fig8vertical_002 | 3.199 | 14.732 | 58.92 | 66/5307 |
| osc_001 | 1.536 | 20.734 | 59.42 | 0/1931 |
| osc_002 | 2.367 | 6.907 | 58.26 | 0/3629 |
| osc_003 | 4.947 | 26.459 | 58.43 | 28/1800 |
| whip1_001 | 1.579 | 10.775 | 61.53 | 54/1484 |
| whip1_002 | 1.032 | 10.672 | 60.62 | 10/1484 |
| whip1_003 | 1.365 | 10.659 | 60.94 | 39/1484 |

Lower marker spread after R^T maps world observations into a rigid local constellation and supports the recorded quaternion direction. It does not establish the aircraft-frame alignment or independently measure the attachment.

Keep 55 mm rigid vertical offset and 63 mm flexible first-span arc length as recorded approximate measurements. Existing fitted lateral offsets remain separate from those measurements.

See geometry.json for both preserved geometry versions, complete descriptive statistics and input hashes.
