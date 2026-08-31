# Milestone 3A — Effective FullState UAV Identification

Date: 2026-08-27  
Command-semantics amendment: 2026-08-28  
Stage: **A — UAV only**  
Stage B EI/Cb: **LOCKED / NOT RUN**  
Stage C joint fitting: **LOCKED / NOT RUN**

## Executive result

The command abstraction was corrected before fitting. The recorded quaternion is used only for yaw/heading; desired roll/pitch comes from the effective translational acceleration plus gravity. The same production `FullStateUAVModel.step` is used by both UAV-only Stage A and the coupled UAV-DDER simulator. EI and Cb stayed fixed and cable loss was zero.

Training used complete takes `osc_001`, `fig8_001`, and `fig8_002`. `fig8_003` was held out from parameter optimization and used only for provisional validation. Because all four takes informed the earlier structural correction, the validation is independent for parameter fitting but not for model-structure selection. A new physical take collected after the model and fitting protocol are frozen is required for completely untouched paper-quality testing.

## Command facts and limitations

- Flight computer/configuration files are external and unavailable by design.
- Firmware is user-confirmed stock/unmodified; exact version has not yet been archived.
- Configuration is user-confirmed default; exact external files have not yet been archived.
- Vehicle type is medium.
- Controller is the user-confirmed default Mellinger controller.
- Estimator is the user-confirmed default Kalman estimator.
- All command-valid `q_cmd` x/y components are exactly zero.
- All command-valid `omega_cmd` values are exactly zero.
- Standard `cmdFullState` defines `omega_cmd` as body-frame rad/s.
- The simulator retains its world-frame angular-velocity state; only the command is interpreted in the body frame.
- Exact external firmware/configuration artifacts have not yet been archived and should be captured when flight-computer access is available.
- Medium vehicle type is metadata only; no mass, inertia, thrust coefficient, or motor parameter enters Stage A.
- This is an effective closed-loop command-response model, not firmware emulation.

## Corrected model

```text
a_eff = K_p (p_cmd - p) + K_v (v_cmd - v) + k_a a_cmd
p_dot = v
v_dot = a_eff

f_des = a_eff + g e_z
b3_des = normalize(f_des)
R_des = geometric_attitude(b3_des, yaw_cmd)

omega_dot = K_R e_R(R_des, R) + K_omega (omega_cmd - omega)
```

No separate gravity term is applied to translational acceleration.

The superseded implementation treated the logged command quaternion as a full
desired roll/pitch/yaw attitude. That structure contradicted the recordings and
was removed before any parameter fitting.

## Fitted parameters

| Parameter | initial | lower | upper | fitted value | provisional-bound hit |
|---|---:|---:|---:|---:|---:|
| K_p | 16.000000 | 1.000000 | 60.000000 | 24.630215 | no |
| K_v | 8.000000 | 0.500000 | 30.000000 | 24.363403 | no |
| k_a | 1.000000 | 0.100000 | 2.000000 | 1.051226 | no |
| K_R | 25.000000 | 1.000000 | 80.000000 | 59.591014 | no |
| K_omega | 10.000000 | 0.500000 | 30.000000 | 8.910099 | no |

Fixed cable values: EI = `2e-06`, Cb = `3.872983346207417e-08`. Cable loss weight = `0`.

## Aggregate results

| Split | takes | windows | position RMSE [m] | position RMSE [mm] | geodesic orientation RMSE [deg] | roll [deg] | pitch [deg] | yaw [deg] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Training | 3 | 321 | 0.0428 | 42.8 | 5.44 | 2.50 | 4.80 | 0.74 |
| Held-out validation | 1 | 188 | 0.0612 | 61.2 | 8.24 | 6.55 | 4.96 | 2.50 |

## Per-take results

| Take | role | windows | position RMSE [m] | orientation [deg] | roll [deg] | pitch [deg] | yaw [deg] |
|---|---|---:|---:|---:|---:|---:|---:|
| fig8_001 | training | 158 | 0.0216 | 4.52 | 1.23 | 4.32 | 0.48 |
| fig8_002 | training | 99 | 0.0339 | 6.05 | 4.01 | 4.48 | 1.10 |
| fig8_003 | validation | 188 | 0.0612 | 8.24 | 6.55 | 4.96 | 2.50 |
| osc_001 | training | 64 | 0.0623 | 5.63 | 1.04 | 5.52 | 0.44 |

## Error versus prediction lead

| Lead [s] | train position [m] | train orientation [deg] | validation position [m] | validation orientation [deg] |
|---:|---:|---:|---:|---:|
| 0.1 | 0.0085 | 2.75 | 0.0122 | 3.69 |
| 0.25 | 0.0241 | 4.87 | 0.0350 | 7.21 |
| 0.5 | 0.0434 | 6.29 | 0.0632 | 9.23 |
| 1.0 | 0.0606 | 5.81 | 0.0847 | 9.46 |

## Optimization

- Bounded Sobol candidates: `32`.
- Adam iterations: `100`.
- Learning rate: `0.02`.
- Seed: `42`.
- Runtime: `231.4 s` on `cuda` using `torch.float64`.
- Hierarchical weighting: time within window, windows within take, then equal mean across training takes.
- Each 1-s prediction is open loop after causal measured initialization; no measured feedback is applied during the window.

## Residual diagnostics

| Take | X position [m] | Y [m] | Z [m] | roll [deg] | pitch [deg] | yaw [deg] |
|---|---:|---:|---:|---:|---:|---:|
| fig8_001 | 0.0127 | 0.0099 | 0.0144 | 1.23 | 4.32 | 0.48 |
| fig8_002 | 0.0244 | 0.0183 | 0.0147 | 4.01 | 4.48 | 1.10 |
| fig8_003 | 0.0460 | 0.0345 | 0.0211 | 6.55 | 4.96 | 2.50 |
| osc_001 | 0.0564 | 0.0155 | 0.0213 | 1.04 | 5.52 | 0.44 |

The per-axis position residuals and roll/pitch residuals remain unequal, so axis anisotropy is present descriptively. It has not been modeled.

| Take | best position lag [s] | best Euler lag [s] |
|---|---:|---:|
| fig8_001 | 0.030 | 0.000 |
| fig8_002 | 0.020 | -0.110 |
| fig8_003 | 0.020 | -0.120 |
| osc_001 | 0.040 | -0.060 |

Positive lag means the measured response occurs later. Position prefers a small, fairly consistent +0.02 to +0.04 s shift; attitude does not show one common positive delay (0.00, -0.11, -0.12, and -0.06 s). This is evidence worth reviewing, not a fitted delay parameter. See `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\reports\milestone3a_stage_a_data\residual_lag_diagnostic.json`.

Held-out overlay: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\reports\milestone3a_stage_a_data\heldout_fig8_003_overlay.png`

## Scientific stop

Stage A is complete. No EI/Cb optimization, joint refinement, drag, downwash, residual network, or cable-reaction model was run. The next decision is whether the held-out error and its structure justify one minimal UAV-model refinement.
