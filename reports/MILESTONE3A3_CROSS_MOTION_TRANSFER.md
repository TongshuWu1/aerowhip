# Milestone 3A.3 — Cross-Motion Transfer and Pre-Test Freeze

Stage B EI/Cb: **LOCKED / NOT RUN**  
Stage C: **LOCKED / NOT RUN**  
MPPI: **NOT RUN**

## Exact model under test

The nominal parameters, 100-ms history, `[e_p, e_v, a_cmd]` feature ordering, `90 -> 32 -> 32 -> 3` SiLU network, normalization procedure, regularization, optimizer, loss, and nominal-only desired-attitude construction are unchanged from authoritative Milestone 3A.2. Each fold trained only residual weights from seed 42. Normalization was recomputed from its two Training motions only.

| Frozen nominal parameter | Value |
|---|---:|
| K_p | 24.630215083 |
| K_v | 24.363402832 |
| k_a | 1.051225526 |
| K_R | 59.591014300 |
| K_omega | 8.910098724 |

The authoritative residual is `100 ms / 10 samples`, with feature order `[e_p_x,e_p_y,e_p_z,e_v_x,e_v_y,e_v_z,a_cmd_x,a_cmd_y,a_cmd_z]`, architecture `90 -> 32 -> SiLU -> 32 -> SiLU -> 3`, and frozen weight hash `8feb4b18ce130641e65fa2bef23801fd3e9b5febfa95277faa550ad32d88c08b`.

## Transfer summary

| Held-out take | Baseline position [mm] | Residual position [mm] | Relative improvement | Baseline orientation [deg] | Residual orientation [deg] |
|---|---:|---:|---:|---:|---:|
| osc_001 | 62.7 | 54.0 | 13.9% | 5.63 | 9.53 |
| fig8_001 | 21.6 | 17.6 | 18.5% | 4.52 | 2.15 |
| fig8_002 | 33.9 | 28.5 | 15.9% | 6.05 | 5.46 |

- Folds with improved position: **3 / 3**.
- Mean relative position improvement: **16.1%**.
- Worst-fold relative position change: **13.9%**.
- Interpretation: **MIXED TRANSFER**.

All three held-out motions improve in position, but Fold 1 has a major orientation degradation. The result therefore fails the no-major-orientation-degradation requirement for GOOD TRANSFER and is classified MIXED TRANSFER.

## Fold 1 — held out `osc_001`

Training takes: `fig8_001`, `fig8_002`.  
Held-out motion: `osc_001`.

| Metric | Physics baseline | Physics + residual |
|---|---:|---:|
| Position RMSE [mm] | 62.7 | 54.0 |
| Orientation RMSE [deg] | 5.63 | 9.53 |
| Residual acceleration RMS [m/s^2] | n/a | 1.763 |
| Residual acceleration p95 [m/s^2] | n/a | 2.611 |

Position by prediction lead:

| Lead [s] | Baseline [mm] | Residual [mm] |
|---:|---:|---:|
| 0.1 | 12.2 | 8.6 |
| 0.25 | 34.9 | 27.4 |
| 0.5 | 63.2 | 53.8 |
| 1.0 | 89.4 | 78.2 |

Position by axis:

| Axis | Baseline [mm] | Residual [mm] |
|---|---:|---:|
| X | 56.9 | 50.9 |
| Y | 15.6 | 16.0 |
| Z | 21.2 | 8.4 |

## Fold 2 — held out `fig8_001`

Training takes: `osc_001`, `fig8_002`.  
Held-out motion: `fig8_001`.

| Metric | Physics baseline | Physics + residual |
|---|---:|---:|
| Position RMSE [mm] | 21.6 | 17.6 |
| Orientation RMSE [deg] | 4.52 | 2.15 |
| Residual acceleration RMS [m/s^2] | n/a | 1.126 |
| Residual acceleration p95 [m/s^2] | n/a | 1.510 |

Position by prediction lead:

| Lead [s] | Baseline [mm] | Residual [mm] |
|---:|---:|---:|
| 0.1 | 4.0 | 2.9 |
| 0.25 | 11.3 | 9.0 |
| 0.5 | 21.0 | 17.3 |
| 1.0 | 32.5 | 26.2 |

Position by axis:

| Axis | Baseline [mm] | Residual [mm] |
|---|---:|---:|
| X | 12.7 | 14.1 |
| Y | 9.9 | 10.2 |
| Z | 14.4 | 2.6 |

## Fold 3 — held out `fig8_002`

Training takes: `osc_001`, `fig8_001`.  
Held-out motion: `fig8_002`.

| Metric | Physics baseline | Physics + residual |
|---|---:|---:|
| Position RMSE [mm] | 33.9 | 28.5 |
| Orientation RMSE [deg] | 6.05 | 5.46 |
| Residual acceleration RMS [m/s^2] | n/a | 1.228 |
| Residual acceleration p95 [m/s^2] | n/a | 2.169 |

Position by prediction lead:

| Lead [s] | Baseline [mm] | Residual [mm] |
|---:|---:|---:|
| 0.1 | 7.2 | 6.0 |
| 0.25 | 20.2 | 17.5 |
| 0.5 | 35.4 | 30.2 |
| 1.0 | 45.7 | 37.8 |

Position by axis:

| Axis | Baseline [mm] | Residual [mm] |
|---|---:|---:|
| X | 24.4 | 22.9 |
| Y | 18.3 | 16.3 |
| Z | 14.7 | 4.7 |

## Existing `fig8_003` context

The authoritative 3A.2 comparison remains approximately `60.6 mm -> 49.9 mm` on the common residual-eligible window set. `fig8_003` was not used by any 3A.3 fold for training, normalization, checkpoint selection, or interpretation tuning.

## Pre-untouched-test freeze

- Freeze: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\model_freezes\UAV_MODEL_FREEZE_PRE_UNTOUCHED_TEST`.
- Authoritative weight SHA-256: `8feb4b18ce130641e65fa2bef23801fd3e9b5febfa95277faa550ad32d88c08b`.
- Normalization canonical SHA-256: `4649a6ba1794ceceab75b1917b09667dafd42f5fa76cc9c288e567a860b8ba95`.
- Evaluation protocol SHA-256: `33d14d40ff285c8398c322fa1c0b79323398ebbbbebde23b8d8016a1b70f51c9`.
- Source snapshot SHA-256: `e6adc2dd223dcd8b0c64e5f51b4cf58067acd3a9d5da916b259f7b25234e774d`.

The freeze is create-once and hash-verified on every later access. The authoritative 3A.2 pointer and weight hash were unchanged by cross-motion training. Fold models remain diagnostics and were not promoted.

## Verification

- `fig8_003` used for fold training/tuning: **NO**.
- Authoritative Milestone 3A.2 weights changed: **NO**.
- Fold 1 post-training evaluator bookkeeping recovery: **YES; saved Training-only checkpoint reused without retraining**.
- Stage B EI/Cb run: **NO**.
- Stage C run: **NO**.
- MPPI run: **NO**.
- Repository tests: **59 passed in 41.40 s**.
- Python compilation: **PASSED**.
- Diff check: **PASSED**.

## Scientific stop

The next physical take must be evaluated exactly once with Model A and the frozen Model C using the frozen protocol. No retraining, normalization recomputation, checkpoint selection, or configuration change is permitted after seeing that take.
