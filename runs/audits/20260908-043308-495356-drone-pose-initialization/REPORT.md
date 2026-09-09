# Drone pose and pre-hover initialization audit

No dynamics fitting or training. Each state is timestamped before CSV playback; no extrapolation to command onset.
The 100 ms preparation margin does not establish hardware synchronization. Attitude change is relative to the measured initial pose, not absolute body tilt.

| Take | Initial speed (m/s) | Attitude change max (deg) | Rigid-offset change max (cm) |
|---|---:|---:|---:|
| whip1_001 | 0.0284 | 39.01 | 3.77 |
| whip1_002 | 0.0142 | 38.13 | 3.68 |
| whip1_003 | 0.0279 | 38.35 | 3.70 |

Commanded hover does not imply zero measured velocity or zero tracking error. No internal controller integral/bias is identified here.
The next implementation step is a nominal P/V/attitude response with explicit command history and hidden-state initialization. Do not fit the old attachment-only surrogate unchanged.
