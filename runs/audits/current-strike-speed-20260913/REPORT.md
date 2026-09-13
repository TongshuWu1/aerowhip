# Current M2 strike-speed check

Compared verified M0/M1/M2 CSVs without changing commands, models or objectives.
The primary comparison replays all three commands under the current M2-selected
model with the same initial state. Evaluation time is the original planned
strike, 1.1172482457473654 s. These are predicted velocities, not impact force.

| Command | Tip speed (m/s) | Forward tip velocity (m/s) | Angle to +x (deg) | Target distance (cm) |
|---|---:|---:|---:|---:|
| M0 | 4.486 | 4.472 | 4.61 | 30.91 |
| M1 | 4.990 | 4.670 | 20.63 | 8.26 |
| M2 | 5.074 | 4.324 | 31.54 | 5.91 |

M1 to M2: forward velocity drops about 7.4%, while total tip speed rises about 1.7%.
The velocity points more upward. Near the closest predicted approach within
0-1.5 s, total speed changes 5.201 to 5.098 m/s (about 2% lower).

Comparing each original saved forecast instead gives 5.224 to 5.074 m/s at
the fixed strike time, about 2.9% lower; those predictions use different models.

The current correction objective contains tip-position, quadrotor-position and
command-position errors. It does not contain explicit tip-velocity or approach-angle
tracking. It therefore permits this directional trade-off while reducing position error.

Previously measured M0/M1 mean tip speeds at the fixed strike time were 5.979 and
5.855 m/s, obtained from local derivatives of measured positions. There are no
M2 physical logs in the current export folder, so no measured M2 speed comparison
is possible. Contact impulse, force and effective impact energy are not established.

The common M2 replay matches its saved tip/cable state to numerical precision;
all CSV and source hashes were verified. See report.json for event definitions,
velocity components, measured-take values and provenance.
