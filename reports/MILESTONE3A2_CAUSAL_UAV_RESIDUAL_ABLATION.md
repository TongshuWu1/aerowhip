# Milestone 3A.2 — Causal UAV Residual Dynamics Ablation

Stage: **UAV translational residual ablation only**  
Stage B EI/Cb: **LOCKED / NOT RUN**  
Stage C joint fitting: **LOCKED / NOT RUN**  
MPPI: **NOT RUN**

## Executive outcome

The accepted five-parameter UAV model was preserved and frozen. The rejected 80-ms fixed-delay model remains an ablation only. A separate optional causal translational residual was added to the same `FullStateUAVModel` and trained using only the three Training takes.

**Model decision: Provisionally adopt Physics + Causal Residual, pending a new untouched physical take.**

This decision is provisional. `fig8_003` was excluded from residual weights, normalization, regularization, and checkpoint selection, but all four current takes influenced earlier model-development decisions. A new untouched take is required before a final generalization claim.

## Frozen nominal model

```text
a_nom = K_p (p_cmd - p_sim) + K_v (v_cmd - v_sim) + k_a a_cmd
p_dot = v
v_dot = a_nom + Delta_a
```

Desired attitude remains constructed from **nominal** `a_nom + g e_z` and commanded yaw. `Delta_a` does not enter desired-attitude construction. There is no angular residual.

| Parameter | authoritative value | status |
|---|---:|---|
| K_p | 24.630215083 | FROZEN |
| K_v | 24.363402832 | FROZEN |
| k_a | 1.051225526 | FROZEN |
| K_R | 59.591014300 | FROZEN |
| K_omega | 8.910098724 | FROZEN |

EI = `2e-06` and Cb = `3.872983346207417e-08` were fixed. Cable loss was zero.

## Residual model

- Input at each sample: `[e_p(3), e_v(3), a_cmd(3)]`.
- History: 10 samples / 100 ms at 100 Hz.
- Flattened input: 90.
- Architecture: `90 -> 32 -> SiLU -> 32 -> SiLU -> 3`.
- Output: `Delta_a_xyz` in m/s^2.
- Trainable parameters: `4067`.
- Final layer initialized exactly to zero, so the initial residual rollout is exactly the accepted baseline.
- No quaternion, angular velocity, cable, take ID, task ID, time index, or measured future state enters the network.

History convention: the initialized FIFO contains `t0-10dt ... t0-dt`. Before each step, the feature built from the current command and current **simulated** UAV state is appended; the oldest sample is dropped; the network therefore sees ten samples ending at the current simulation time. Only pre-window initialization uses measured causal position/velocity history.

## Training-only normalization

Statistics were computed from accepted nominal-baseline rollouts over Training takes only. Validation was not read.

| Feature | mean | standard deviation used |
|---|---:|---:|
| position_error_x | -0.003604 | 0.027779 |
| position_error_y | 0.000112 | 0.011670 |
| position_error_z | -0.024332 | 0.009026 |
| velocity_error_x | 0.003556 | 0.030423 |
| velocity_error_y | -0.000108 | 0.016667 |
| velocity_error_z | 0.023969 | 0.009289 |
| command_acceleration_x | 0.009944 | 1.231214 |
| command_acceleration_y | -0.008520 | 0.980687 |
| command_acceleration_z | 0.000000 | 0.001000 |

Standard-deviation floor: `0.001`.  
Normalization hash: `4649a6ba1794ceceab75b1917b09667dafd42f5fa76cc9c288e567a860b8ba95`.

## Training procedure

- Optimized values: residual-network weights `phi` only.
- Optimizer: Adam.
- Seed: `42`.
- Learning rate: `0.001`.
- Updates: `500`.
- Windows per physical take per update: `32`.
- Full-training checkpoint evaluation interval: `25` updates.
- Gradient clip: `5.0`.
- Runtime: `1116.4 s` on `cuda` using `torch.float64`.
- Best checkpoint was selected only by the complete hierarchical Training objective.
- `fig8_003` was evaluated only after training/checkpoint selection completed.

One conservative predeclared regularization setting was used; no hyperparameter search was performed:

```text
lambda_magnitude  = 0.01
lambda_smoothness = 0.05
acceleration scale = 1.0 m/s^2
```

## Three-model aggregate comparison

| Split | Model | position RMSE [mm] | orientation RMSE [deg] |
|---|---|---:|---:|
| Training | Physics baseline | 42.8 | 5.44 |
| Training | Fixed-delay ablation (rejected) | 30.5 | 5.21 |
| Training | Physics + causal residual | 30.4 | 4.65 |
| Validation | Physics baseline | 61.2 | 8.24 |
| Validation | Fixed-delay ablation (rejected) | 44.6 | 7.43 |
| Validation | Physics + causal residual | 49.9 | 7.46 |

The fixed-delay model is included only as a diagnostic. It remains rejected because tau reached the 80-ms upper bound, its objective remained decreasing there, gain-delay confounding was substantial, and no supported finite interior delay was identified.

## Held-out prediction-lead comparison

Each cell is `position RMSE [mm] / orientation RMSE [deg]`.

| Lead [s] | Physics baseline | Rejected fixed delay | Physics + residual |
|---:|---:|---:|---:|
| 0.1 | 12.2 / 3.69 | 6.9 / 3.09 | 8.9 / 4.30 |
| 0.25 | 35.0 / 7.21 | 24.7 / 6.78 | 28.5 / 8.37 |
| 0.5 | 63.2 / 9.23 | 47.2 / 8.35 | 52.9 / 8.12 |
| 1.0 | 84.7 / 9.46 | 60.9 / 8.35 | 67.1 / 7.92 |

Position prediction improves at all four held-out leads. Orientation does not improve uniformly: the residual is worse at 0.10 s and 0.25 s, but better at 0.50 s and 1.00 s, with a lower aggregate validation orientation RMSE. This is consistent with the residual being trained as a translational correction while desired attitude remains nominal; no angular-residual claim is made.

## Per-take comparison

| Take | role | residual windows | baseline position [mm] | delay position [mm] | residual position [mm] | baseline orientation [deg] | delay orientation [deg] | residual orientation [deg] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fig8_001 | training | 158 | 21.6 | 22.7 | 15.6 | 4.52 | 4.55 | 2.21 |
| fig8_002 | training | 99 | 33.9 | 29.5 | 24.5 | 6.05 | 5.70 | 3.93 |
| fig8_003 | validation | 184 | 60.6 | 44.6 | 49.9 | 8.05 | 7.43 | 7.46 |
| osc_001 | training | 63 | 62.7 | 37.4 | 43.9 | 5.63 | 5.32 | 6.67 |

The additional causal-history gate rejected one `osc_001` window and four `fig8_003` windows because they lacked a complete valid 100-ms residual history plus the accepted causal velocity-estimation history inside their usable segments. No segment annotation was changed. Baseline-versus-residual acceptance metrics use the identical 320 Training and 184 validation windows; the fixed-delay column is retained as its already-published diagnostic artifact and used 321/188 windows.

## Axis and Euler diagnostics

Each cell is `physics baseline / physics + residual`.

| Take | X [mm] | Y [mm] | Z [mm] | roll [deg] | pitch [deg] | yaw [deg] |
|---|---:|---:|---:|---:|---:|---:|
| fig8_001 | 12.7 / 12.1 | 9.9 / 9.6 | 14.4 / 2.5 | 1.23 / 1.32 | 4.32 / 1.74 | 0.48 / 0.39 |
| fig8_002 | 24.4 / 19.3 | 18.3 / 14.6 | 14.7 / 3.6 | 4.01 / 2.59 | 4.48 / 2.89 | 1.10 / 0.88 |
| fig8_003 | 45.5 / 41.5 | 34.1 / 26.8 | 21.0 / 7.2 | 6.32 / 4.97 | 4.94 / 5.40 | 2.46 / 2.17 |
| osc_001 | 56.9 / 41.3 | 15.6 / 11.4 | 21.2 / 9.6 | 1.05 / 1.95 | 5.52 / 6.38 | 0.45 / 0.79 |

## Residual statistics

| Take | role | RMS norm | X RMS | Y RMS | Z RMS | median norm | p95 norm | maximum norm | smoothness RMS | smoothness/RMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fig8_001 | training | 1.011 | 0.589 | 0.094 | 0.816 | 0.966 | 1.334 | 2.412 | 0.044 | 0.044 |
| fig8_002 | training | 1.290 | 0.703 | 0.684 | 0.837 | 0.949 | 2.487 | 3.762 | 0.097 | 0.076 |
| fig8_003 | validation | 2.053 | 1.131 | 1.324 | 1.087 | 1.289 | 4.120 | 6.874 | 0.186 | 0.091 |
| osc_001 | training | 1.576 | 1.196 | 0.296 | 0.983 | 1.481 | 2.475 | 3.552 | 0.110 | 0.070 |

All acceleration values are m/s^2. Smoothness is the RMS norm of `Delta_a[t]-Delta_a[t-1]`.

The held-out residual RMS is `2.053 m/s^2`, compared with `1.313 m/s^2` across Training takes. Its p95 is `4.120 m/s^2` and maximum is `6.874 m/s^2`. These remain inside the predeclared plausibility gates, but the larger held-out residual magnitude is a real distribution-shift warning and reinforces the requirement for a new untouched take.

### Acceleration onset and reversal

| Take | onset count | mean residual norm at onset | reversal count | mean residual norm at reversal | aggressive-turn count | mean residual norm at turn |
|---|---:|---:|---:|---:|---:|---:|
| fig8_001 | 32 | 1.069 | 0 | n/a | 7 | 0.995 |
| fig8_002 | 30 | 1.431 | 15 | 1.605 | 40 | 1.391 |
| fig8_003 | 59 | 2.096 | 29 | 2.668 | 66 | 2.354 |
| osc_001 | 39 | 1.447 | 3 | 1.891 | 3 | 1.891 |

Reversal and aggressive-turn events compare command-acceleration direction over the preceding 100 ms; an aggressive turn is a newly entered direction change above 30 degrees while both endpoints are excited. These diagnostics show whether the residual responds systematically to command transients. They are not interpreted as identified physical forces or explicit delay.

## Acceptance gates

| Gate | Result |
|---|---|
| all rollouts finite | PASS |
| more than one training motion improved | PASS |
| multiple validation horizons improved | PASS |
| no validation axis materially degraded | PASS |
| residual maximum plausible | PASS |
| residual p95 plausible | PASS |
| residual temporally smooth | PASS |
| validation orientation not materially degraded | PASS |
| validation position materially improved | PASS |

- Held-out position relative improvement: `17.6%`.
- Improved held-out horizons: `['0.1', '0.25', '0.5', '1.0']`.
- Improved Training takes: `['fig8_001', 'fig8_002', 'osc_001']`.
- Held-out orientation ratio residual/baseline: `0.926`.
- Held-out axis RMSE ratios residual/baseline: `{'x': 0.9119828299601623, 'y': 0.7883369141849568, 'z': 0.3403620041707374}`.
- Decision: **`PROVISIONALLY_ADOPT_PHYSICS_PLUS_CAUSAL_RESIDUAL`**.

## Causal-history window audit

| Take | accepted | rejected from original Stage A set |
|---|---:|---:|
| fig8_001 | 158 | 0 |
| fig8_002 | 99 | 0 |
| fig8_003 | 184 | 4 |
| osc_001 | 63 | 1 |

Exact rejection reasons and indices are stored in `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T124719.940407+0000_4c7577a0\residual_window_audit.json`.

## Production integration

The residual is an optional modular component of the existing `FullStateUAVModel` and `CoupledSimulator`. The public `reset`, `step`, and `rollout` APIs remain in use. Residual history is carried in `UAVState`; there is no hidden Python-side rollout history. The path supports batching, CUDA, autograd, deterministic reset, and exact state-dict reload. With the residual disabled, the existing baseline arithmetic path is unchanged. With its zero-initialized final layer enabled, trajectories are exactly equal to baseline.

## Reproducibility

- Artifact: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T124719.940407+0000_4c7577a0`.
- Model description: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T124719.940407+0000_4c7577a0\model.json`.
- Weights: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T124719.940407+0000_4c7577a0\uav_residual.pt`.
- Weight SHA-256: `8feb4b18ce130641e65fa2bef23801fd3e9b5febfa95277faa550ad32d88c08b`.
- Normalization: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T124719.940407+0000_4c7577a0\normalization.json`.
- Normalization canonical SHA-256: `4649a6ba1794ceceab75b1917b09667dafd42f5fa76cc9c288e567a860b8ba95`.
- Dataset snapshot: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T124719.940407+0000_4c7577a0\dataset_snapshot.json`.
- Training history: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T124719.940407+0000_4c7577a0\training_history.csv`.
- Training metrics: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T124719.940407+0000_4c7577a0\training_metrics.csv`.
- Validation metrics: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T124719.940407+0000_4c7577a0\validation_metrics.csv`.
- Per-take metrics: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T124719.940407+0000_4c7577a0\per_take_metrics.csv`.
- Residual statistics: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T124719.940407+0000_4c7577a0\residual_statistics.csv`.
- Verification status: `PASSED`.
- Repository tests: `{'passed': 56, 'failed': 0, 'command': '.venv\\Scripts\\python.exe -m pytest -q'}`.
- Baseline saved-replay maximum position/orientation difference: `{'position_m': 0.0, 'orientation_quaternion_component': 0.0}`.
- Reloaded residual saved-replay maximum differences: `{'position_m': 0.0, 'orientation_quaternion_component': 0.0, 'residual_acceleration_mps2': 0.0}`.

## Scientific stop

No EI/Cb fitting, cable residual, fixed delay, axis-specific gain, nominal-parameter refit, angular residual, drag model, MPPI, or Stage C joint refinement was run. The result must be reviewed before freezing the UAV boundary model or beginning cable identification.
