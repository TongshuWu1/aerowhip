# Whip targeting accuracy

Closest measured cable-tip distance to the intended target centre (1.25, 0, 1.25) m, over the executed command whip, 0–1.133333333 s. The long braking/recovery tail is excluded. Prediction RMSE is a separate metric.

| Model | Closest distance, mean ± SD (cm) | Within 2 cm | Within 5 cm |
|---|---:|---:|---:|
| M0 | 12.58 ± 1.41 | 0/11 | 0/11 |
| M1 | 4.86 ± 1.76 | 1/12 | 7/12 |
| M2 | 4.64 ± 2.47 | 2/11 | 7/11 |
| M3 | 4.71 ± 1.81 | 0/12 | 7/12 |
| M4 | 3.81 ± 2.04 | 3/12 | 7/12 |

No physical target was present. Threshold counts describe estimated geometric proximity, not confirmed contacts. Distances use linear interpolation between adjacent valid native OptiTrack samples; gaps are not bridged. Prediction evaluation still ends at the fixed planned strike (1.113800079 s). This targeting metric allows closest approach anywhere during the executed whip.
