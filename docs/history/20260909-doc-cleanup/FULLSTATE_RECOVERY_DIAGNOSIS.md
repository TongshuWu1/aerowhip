> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Supplied full-state CSV: recovery transition diagnosis

Follow-up: a separate gentle-recovery CSV and updated UI export are now implemented. See [current export workflow](FULLSTATE_TESTING.md). The diagnosis below describes the original supplied CSV; its files remain unchanged. The new reference removes the abrupt recovery reversal but has not been validated on the vehicle.

Audited `C:/Users/wts28/Downloads/fullstate_30hz.csv` on 7 September 2026. Its 613 rows exactly match `runs/rehearsals/20260907-221528-727626/plan_001/fullstate_30hz.csv` (SHA256 `0f33bc1522658ba961e9ec905ac3f9f5e7bc9da0201732120497c428dad267ef`). The recorded reference lasts 20.4 s with a 0.8 s whip. No CSV, policy, controller, logger or active simulation was changed.

## What the reference demands

| Time (s) | CSV phase | ax (m/s²) | az (m/s²) | Signed XZ feedforward direction from +Z |
|---|---|---|---|---|
| 0.800 | whip boundary | -5.690 | -5.582 | -53.41° |
| 0.83333 | PID recovery | +6.649 | -6.416 | +62.98° |

Using `a + [0,0,g]` as a feedforward thrust-direction surrogate, this is a 116.39° direction change in 33.33 ms, or an average reference rate of 3491.8°/s. This is not measured attitude or a verified vehicle limit. The calculation ignores cable loading, position/velocity feedback and tracked-point offsets. The whip itself reaches a 73.11° surrogate tilt, so the entire trajectory needs assessment, even if the intended code change affects only recovery.

The source simulator also changes its actual commanded force from `[-1.60786, -0.00480, 0.75538]` N to `[0.78370, -0.00923, 0.79031]` N at the controller transition, a 109.59° direction jump. Therefore the abrupt reversal is present before CSV interpolation. The point-mass PID limits force norm and clamps Fz nonnegative, but has no attitude state, tilt bound or thrust-direction rate limit. Exporting consistent derivatives alone does not make that trajectory dynamically trackable.

Negative az alone is not evidence of impossibility. Recovery minimum az is -6.416 m/s²; all CSV rows have az+g positive. However, simultaneous horizontal braking and downward acceleration ask for a strongly tilted, reduced-vertical-support thrust direction.

## Plausible mechanism for the observed drop

The [upstream Bitcraze Mellinger controller](https://github.com/bitcraze/crazyflie-firmware/blob/master/src/modules/src/controller/controller_mellinger.c) forms desired force from acceleration plus gravity and feedback, then projects it onto the current body thrust axis. If the vehicle still points along the previous reference direction, the next feedforward vector here has a negative projection (-3.318 m/s² in mass-normalized units). This can sharply reduce collective thrust during the reversal. This is a hypothesis consistent with the CSV, not confirmation of what happened onboard. The actual firmware build, attitude, tracking errors and thrust/motor logs must establish the real mechanism.

## Required recovery change

Generate recovery from the terminal position, velocity and acceleration/thrust direction of the whip, with continuous acceleration and bounded changes in thrust direction. Slow braking/return as needed, enforce verified thrust, tilt, rotation-rate/jerk and workspace limits, and derive position, velocity and acceleration from the same trajectory. Check stopping distance and altitude throughout the transition, including between CSV samples. Include cable loading or a justified allowance in the thrust feasibility test. A pure point-mass replay cannot certify attitude tracking.

Do not repair this by clipping negative acceleration in the CSV: unchanged position and velocity would then contradict acceleration. Do not silently change the whip to satisfy recovery limits. First assess whether its terminal state permits a feasible recovery within the available space; if not, the whip itself must be revised explicitly. Keep full-state control through the complete sequence.

Numerical limits are not available in the saved export settings. The saved 3.2 N simulation force cap is not a verified vehicle capability. No new recovery implementation or flight-ready trajectory was produced in this diagnostic step.

Reproduce the read-only audit with `.venv/Scripts/python.exe runs/audits/20260907-fullstate-recovery/audit.py`. Results and a reference/source comparison plot are in that directory. Original files remain unchanged.
