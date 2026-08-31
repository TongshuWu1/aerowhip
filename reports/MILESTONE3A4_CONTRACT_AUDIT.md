# Milestone 3A.4 Command and Timeline Contract Audit

Date: 2026-08-28  
Scope: audit only, completed before GUI or scientific-workflow changes.

## Gate result

| Contract | Result | Consequence |
|---|---|---|
| Crazyflie command semantics | **PASS** | Metadata/documentation correction only; no retraining |
| Motive manual-trim timeline authority | **PASS** | Metadata enrichment only; no numerical reprocessing required |

Neither scientific stop condition was triggered. The accepted Milestone 3A.2/3A.3 numerical implementation already had the correct command abstraction, and every authoritative processed take already contained exactly the Motive-export samples.

## A. Command semantics audit — PASS

The executed production path is [`simulator/uav/model.py`](../simulator/uav/model.py):

1. `FullStateUAVModel.step` reads position, velocity and acceleration explicitly.
2. It evaluates

   `a_nom = k_a a_cmd + K_p(p_cmd-p) + K_v(v_cmd-v)`.

3. `q_cmd` enters only through `yaw_from_quaternion_xyzw(command_heading)`.
4. [`desired_rotation_from_acceleration_and_yaw`](../simulator/uav/quaternion.py) constructs body Z from `a_nom + g e_z` and completes the desired attitude with commanded yaw.
5. Commanded roll/pitch are never read from `q_cmd.x/y`.
6. The generic command angular velocity is interpreted in the body frame. Every valid current-data value is exactly zero.
7. The accepted residual modifies realized translation only; desired attitude continues to use nominal `a_nom`.

Recorded-data evidence:

| Take | max ‖q_cmd,xy‖ | max ‖omega_cmd‖ | max horizontal a_cmd [m/s²] |
|---|---:|---:|---:|
| osc_001 | 0 | 0 | 4.285 |
| fig8_001 | 0 | 0 | 1.997 |
| fig8_002 | 0 | 0 | 4.494 |
| fig8_003 | 0 | 0 | 9.024 |

The scientific command is therefore encoded as externally supplied `p / v / a / yaw`. The quaternion is retained as a yaw-derived provenance field, not a full external attitude command. Omega is retained for API provenance/future compatibility and is documented as unexcited in the current experiments.

Supplied experiment provenance is now explicit:

- controller: Mellinger;
- estimator: Kalman;
- robot type: `bolt_3in_2s`;
- motion capture: vendor tracking;
- firmware: stock/unmodified; exact version not archived.

This provenance does not add firmware, motor, inertia, thrust, or controller-internal parameters to the effective model.

## B. Timeline contract audit — PASS

The authoritative processor constructs every stored array on `motive.source_time_s` and sets:

`time_s = motive.source_time_s - motive.source_time_s[0]`.

Commands are reconstructed by zero-order hold at those Motive observation times. Logger samples outside the Motive interval cannot add scientific frames, extend duration, enter playback, or form fitting-window endpoints.

The audit compared all three stored identity arrays—`time_s`, `motive_source_time_s`, and `motive_frame`—against a fresh parse of each manually trimmed Motive CSV. Equality was bit-for-bit, not approximate.

| Take | Motive first frame / source time [s] | Motive last frame / source time [s] | Motive duration [s] | logger synchronized first / last [s] | processed start / end [s] | command coverage | exact Motive identity |
|---|---:|---:|---:|---:|---:|---:|---|
| osc_001 | 1274 / 12.74 | 3204 / 32.04 | 19.30 | 1.55 / 39.51 | 0.00 / 19.30 | 92.49% | PASS, max Δt = 0 |
| fig8_001 | 639 / 6.39 | 4923 / 49.23 | 42.84 | -3.88 / 56.05 | 0.00 / 42.84 | 94.52% | PASS, max Δt = 0 |
| fig8_002 | 949 / 9.49 | 3983 / 39.83 | 30.34 | 0.12 / 53.06 | 0.00 / 30.34 | 91.73% | PASS, max Δt = 0 |
| fig8_003 | 958 / 9.58 | 6687 / 66.87 | 57.29 | -0.43 / 83.29 | 0.00 / 57.29 | 98.46% | PASS, max Δt = 0 |

Available synchronized logger pre-history before the Motive boundary is respectively 11.19, 10.27, 9.37 and 10.01 seconds. It remains causal initialization provenance only. Logger post-history—7.47, 6.82, 13.23 and 16.42 seconds respectively—is explicitly excluded.

## C. Freeze-integrity decision

The timeline and command audits found no scientific implementation bug. The pre-untouched-test freeze remains valid.

- processed NPZ hashes before/after metadata enrichment: unchanged for all four takes;
- frozen residual weight SHA-256: `8feb4b18ce130641e65fa2bef23801fd3e9b5febfa95277faa550ad32d88c08b`;
- normalization canonical SHA-256: `4649a6ba1794ceceab75b1917b09667dafd42f5fa76cc9c288e567a860b8ba95`;
- evaluation protocol canonical SHA-256: `33d14d40ff285c8398c322fa1c0b79323398ebbbbebde23b8d8016a1b70f51c9`.

No model was retrained. No gain, residual weight, normalization value, architecture, fitting window, EI, Cb, DDER equation, or accepted numerical result was changed.

