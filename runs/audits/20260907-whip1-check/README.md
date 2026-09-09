# Whip1 inspection — 7 September 2026

User provenance: these are 6 September flights using a precomputed open-loop
full-state reference produced by executing PPO in simulation, then tracking
with cmdFullState. No policy was executed onboard. This check changes no
simulator, controller, policy, calibration, or Drive file.

Sources: https://drive.google.com/drive/folders/19WifpxNU8EON1y2f4k41j2I1Z9WVrqRh
Files named `experiment_whip1_001/002/003.csv` are controller-side logs;
`whip1_001/002/003.csv` are hand-trimmed OptiTrack exports. Original bytes and
SHA-256 hashes are preserved in `raw/` and `results.json`.

All three exports contain 1484 frames spanning original take times 6.24–21.07 s,
100 Hz, global coordinates in meters, and rigid body cf_7. No cf_7 position
rows are missing. Capture dates say January 1970 and are unsuitable for
absolute wall-clock alignment. Correction after full-header inspection: all three files contain the named
marker set `cable1`, with XYZ positions for `c1` through `c10`, in addition
to unlabeled tracks. These named columns were missed in the initial summary.
CSV column order is c1, c10, c2, ..., c9; use numeric marker order when
extracting. No cable reconstruction or strike scoring has yet been performed.

Matching measured XYZ with one constant time shift, without spatial fitting:

| Pair | controller time = OptiTrack time + offset | 3D RMS disagreement |
|---|---:|---:|
| 001 | +1.1752 s | 1.11 cm |
| 002 | −0.5346 s | 1.13 cm |
| 003 | +0.0355 s | 0.96 cm |

These are empirical alignment offsets including logging latency, not hardware
clock offsets. The RMS measures agreement between measured logs, not desired
trajectory tracking accuracy. Split-half checks are in results.json.

All three controller logs contain exactly the first 20 p/v/a reference samples
from `runs/rehearsals/20260906-222405-804438/plan_001/fullstate_30hz.csv`, with
zero numeric difference. Samples cover reference times 0 through 0.633333 s.
At approximately 0.667–0.669 s from the first maneuver command, the log changes
to the then-measured XYZ position, with zero desired velocity/acceleration.
Reference times 0.666667, 0.700000, 0.733333, 0.766667 and the 0.800000 terminal
boundary are absent. This establishes a shorter recorded execution, not why
the executor chose to stop. The reference tail includes braking commands.

Controller rows are approximately 100 Hz, but logged measured XYZ changes at
approximately 10 Hz. Recorded velocities include repeated zeros and spikes;
they should not be interpreted as reliable instantaneous vehicle velocities.
This observation does not establish the onboard estimator/controller rate.

OptiTrack altitude peaks are 2.464, 2.466 and 2.435 m across the retained windows.
The reference intentionally rises, and the subsequent fixed-position commands
hold elevated measured positions. An overlay is provided in alignment.png.
No cause of the tracking discrepancy, cable-model fit, or success claim is
established by this initial inspection. Next relevant evidence is the exact
executor script/configuration and its stopping logic, plus the logger's position
subscription/source rate.
