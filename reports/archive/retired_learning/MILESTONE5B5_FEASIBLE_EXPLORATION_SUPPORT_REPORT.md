# Milestone 5B.5 — Feasible Exploration Support Report

## 1. Why 5B.4 changes the diagnosis

5B.4 sampled 99.3% progress and 9.38-mm proximity yet maintained 0% stochastic feasibility, with finite but physically impossible speeds and displacements. That changes the immediate question from reward/entropy design to simulator-validity root cause and feasible action-support geometry.

## 2. No learning performed

No SAC, actor, critic, alpha, behavior-cloning, CEM, or policy-fitting update occurred. All actor checkpoints were read-only sources of diagnostic actions/centers.

## 3. Frozen production model

`MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`, production CUDA float32 PCG32 DDER, residual, geometry, gains, three substeps, four projections, reward-v2, hard gates, canonical state/target/direction, and T=1.20 s were unchanged. The only altered model instance was separately labeled `RESIDUAL_DISABLED_DIAGNOSTIC_ONLY` and was never used for production results.

## 4. Recoverability and selected 5B.4 trajectory classes

5B.4 exact stochastic replay actions were **not persisted**: `{"checkpoint_contains_replay_metadata_only": true, "consequence": "historical 5B.4 stochastic outliers cannot be exactly replayed; a seeded frozen-checkpoint diagnostic batch is used", "exact_historical_stochastic_actions_recoverable": false, "replay_buffer_file_present": false, "source_artifact": "C:\\Users\\wts28\\Documents\\PHD\\particle_filter_cable_project\\data\\policy_training\\sac_canonical_spectral_entropy_v1\\2026-08-30T015034.715379Z", "trajectory_storage_flag": false}`. Consequently historical 5B.4 outliers cannot be honestly replayed exactly. A reproducible replacement diagnostic batch of 2,048 actions was sampled once from the frozen 5B.4 latest checkpoint with seed 5504; the 12 selected tensors were then saved exactly and never regenerated. Groups contain three mild, three high-progress, three near-target, and three catastrophic actions without duplication.

## 5. Batch-one reproduction

The same newly selected tensors were compared between their original B=2,048 diagnostic batch and batch-one replay with the fixed-UAV evaluation shape 2,048. 12/12 passed the defined classification/metric equivalence; 0 were flagged `NUMERICAL_BATCH_REPRODUCTION_ISSUE`. This gate pertains to the new diagnostic actions, not unrecoverable historical actions.

## 6. Command-consistency verification

Command generation: **HEALTHY**. Every selected FullState command remained finite, used the exact piecewise-linear acceleration integration, stayed at or below 20 m/s², and passed discrete p/v/a consistency. Across the 12 actions, maxima were 19.887852 m/s² commanded acceleration, 7.586700 m/s commanded velocity, and 5.476614 m commanded displacement; maximum discrete consistency errors were 1.788e-06 m/s for velocity and 8.047e-07 m for position. Thus command construction is correct, although some mathematically valid sampled commands are far outside the vehicle's feasible displacement/speed envelope.

## 7. Catastrophic trace and first runaway

Representative action: `D_CATASTROPHIC_1`. Its command peaks at 19.262 m/s², 7.318 m/s, and 5.477 m. At t=0, b3_des=[0.9959779381752014, -0.00860876590013504, 0.08918449282646179] is nearly horizontal, while realized acceleration before the residual is [0.08308324217796326, -0.006565779447555542, -3.6006717681884766] m/s² and residual acceleration is only [0.0160320196300745, -0.34495383501052856, -0.26562607288360596] m/s². Tracking mismatch therefore starts immediately.

Production causal sequence: normalized residual |z|>10 is already present at t=0; residual ||Delta_a|| first exceeds 10 m/s² at 0.19 s; UAV speed exceeds the 3 m/s feasibility limit at 0.20 s; controller acceleration exceeds 100 m/s² at 0.23 s; total acceleration exceeds 100 m/s² at 0.25 s; UAV speed exceeds 10 m/s at 0.28 s. The predeclared runaway detector fires at 0.23 s. Angular velocity never exceeds 100 rad/s.

With the residual disabled, the sequence is earlier: UAV speed >3 m/s at 0.17 s, controller acceleration >100 m/s² at 0.20 s, total acceleration >100 m/s² at 0.21 s, and UAV speed >10 m/s at 0.25 s. The full seven-step causal window around production onset is preserved in `runaway_root_cause.json` and the trace figures.

## 8. Residual-input OOD analysis

Across selected production traces, maximum normalized residual feature magnitude was 1.13e+07. Aggregate timestep fractions with any |z|>3, >5, and >10 were 100.00%, 100.00%, and 99.51%. Residual OOD observed: **YES**. For the representative trace, max-|z| and ||Delta_a|| correlation is 0.997019; however, OOD is present from t=0, Delta_a is initially only 0.436 m/s², and the residual-off counterfactual runs away sooner and farther. OOD correlates with the diverging state but is neither necessary nor dominant for this runaway.

## 9. Residual-off counterfactual

For the representative action, production max UAV speed was 3.11e+05 m/s versus 7.91e+05 m/s with Delta_a identically zero. Residual-off status: **NO**. All counterfactuals are separately labeled diagnostic-only.

## 10. Root-cause classification

**CLOSED_LOOP_TRACKING_RUNAWAY**. Tracking-error feedback grows first and runaway persists with the residual disabled. The initiating condition is a dynamically infeasible but correctly integrated aggressive command: attitude/thrust realization lags the desired force direction, position/velocity error grows, and unsaturated Kp/Kv feedback magnifies the demanded acceleration. No angular-state explosion precedes the translational runaway. Residual extrapolation is a secondary validity concern, not the necessary root cause demonstrated here.

## 11. Production model not modified

No clamp, saturation, residual retraining/removal, controller change, or physics modification was implemented. The evidence is returned before any remedy.

## 12. Feasible-support audit design

The audit used exact production physics for 30,720 rollouts: centers ZERO and 5B4_LEVEL1_SAC; active spectral bandwidth K=4,8,16; noise scales 0.05,0.10,0.20,0.35,0.50; 1,024 actions/cell; fixed T=1.20 s; IDCT, radial squash, physical decoder, and FullState integration unchanged.

## 13. Exact centers and reconstruction

CENTER A is zero raw spectral acceleration. CENTER B is the saved best 5B.4 deterministic actor action, not CEM. Its inverse-radial/DCT/IDCT/radial maximum normalized-action reconstruction error was 3.446e-08; status **PASS**.

## 14. Full support table

| Center | K | Scale | Feasible | Feas.&P>=.25 | Feas.&P>=.50 | P95 progress | Min d (mm) | Successes | UAV speed p99/max | Classification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ZERO | 4 | 0.05 | 88.48% | 3.03% | 0.00% | 0.261 | 705.3 | 0 | 1.01/1.21 | USEFUL_SUPPORT |
| ZERO | 4 | 0.10 | 32.13% | 4.10% | 0.10% | 0.456 | 251.7 | 0 | 2.04/2.42 | USEFUL_SUPPORT |
| ZERO | 4 | 0.20 | 5.27% | 1.46% | 0.39% | 0.601 | 15.0 | 0 | 4.07/4.69 | UNSAFE_SUPPORT |
| ZERO | 4 | 0.35 | 0.39% | 0.10% | 0.00% | 0.659 | 12.0 | 0 | 204.57/1318.70 | UNSAFE_SUPPORT |
| ZERO | 4 | 0.50 | 0.10% | 0.10% | 0.00% | 0.697 | 60.5 | 0 | 1829.76/49701.19 | UNSAFE_SUPPORT |
| ZERO | 8 | 0.05 | 88.67% | 4.39% | 0.00% | 0.270 | 572.0 | 0 | 1.10/1.32 | USEFUL_SUPPORT |
| ZERO | 8 | 0.10 | 31.15% | 3.52% | 0.10% | 0.383 | 347.8 | 0 | 2.12/2.44 | USEFUL_SUPPORT |
| ZERO | 8 | 0.20 | 5.47% | 1.56% | 0.10% | 0.606 | 84.7 | 0 | 4.74/79.32 | UNSAFE_SUPPORT |
| ZERO | 8 | 0.35 | 0.59% | 0.10% | 0.00% | 0.679 | 31.9 | 0 | 954.56/6399.77 | UNSAFE_SUPPORT |
| ZERO | 8 | 0.50 | 0.00% | 0.00% | 0.00% | 0.663 | 88.5 | 0 | 4459.42/52161.42 | UNSAFE_SUPPORT |
| ZERO | 16 | 0.05 | 88.67% | 3.52% | 0.00% | 0.268 | 708.8 | 0 | 1.05/1.30 | USEFUL_SUPPORT |
| ZERO | 16 | 0.10 | 33.11% | 4.49% | 0.20% | 0.425 | 297.2 | 0 | 2.10/2.71 | USEFUL_SUPPORT |
| ZERO | 16 | 0.20 | 5.18% | 0.98% | 0.10% | 0.606 | 72.1 | 0 | 5.06/78.80 | UNSAFE_SUPPORT |
| ZERO | 16 | 0.35 | 0.39% | 0.29% | 0.00% | 0.678 | 91.5 | 0 | 1508.88/5865.88 | UNSAFE_SUPPORT |
| ZERO | 16 | 0.50 | 0.10% | 0.10% | 0.10% | 0.600 | 71.7 | 0 | 11913.95/103596.42 | UNSAFE_SUPPORT |
| 5B4_LEVEL1_SAC | 4 | 0.05 | 51.07% | 19.14% | 0.00% | 0.506 | 334.0 | 0 | 1.31/1.54 | USEFUL_SUPPORT |
| 5B4_LEVEL1_SAC | 4 | 0.10 | 22.66% | 5.96% | 0.20% | 0.613 | 131.9 | 0 | 2.27/2.54 | USEFUL_SUPPORT |
| 5B4_LEVEL1_SAC | 4 | 0.20 | 5.66% | 1.76% | 0.20% | 0.707 | 29.5 | 0 | 4.08/6.43 | UNSAFE_SUPPORT |
| 5B4_LEVEL1_SAC | 4 | 0.35 | 0.10% | 0.00% | 0.00% | 0.723 | 53.7 | 0 | 238.55/1877.52 | UNSAFE_SUPPORT |
| 5B4_LEVEL1_SAC | 4 | 0.50 | 0.00% | 0.00% | 0.00% | 0.756 | 56.9 | 0 | 1820.96/55067.47 | UNSAFE_SUPPORT |
| 5B4_LEVEL1_SAC | 8 | 0.05 | 49.90% | 17.09% | 0.00% | 0.517 | 319.2 | 0 | 1.30/1.56 | USEFUL_SUPPORT |
| 5B4_LEVEL1_SAC | 8 | 0.10 | 23.05% | 6.45% | 0.10% | 0.639 | 72.6 | 0 | 2.20/2.67 | USEFUL_SUPPORT |
| 5B4_LEVEL1_SAC | 8 | 0.20 | 4.39% | 1.56% | 0.39% | 0.711 | 50.1 | 0 | 4.53/77.86 | UNSAFE_SUPPORT |
| 5B4_LEVEL1_SAC | 8 | 0.35 | 0.68% | 0.39% | 0.20% | 0.713 | 41.4 | 0 | 800.07/7458.08 | UNSAFE_SUPPORT |
| 5B4_LEVEL1_SAC | 8 | 0.50 | 0.10% | 0.00% | 0.00% | 0.696 | 16.5 | 0 | 7130.40/52176.97 | UNSAFE_SUPPORT |
| 5B4_LEVEL1_SAC | 16 | 0.05 | 50.78% | 16.02% | 0.00% | 0.499 | 376.9 | 0 | 1.28/1.49 | USEFUL_SUPPORT |
| 5B4_LEVEL1_SAC | 16 | 0.10 | 21.39% | 6.05% | 0.29% | 0.633 | 104.3 | 0 | 2.32/2.80 | USEFUL_SUPPORT |
| 5B4_LEVEL1_SAC | 16 | 0.20 | 5.86% | 1.95% | 0.49% | 0.746 | 89.0 | 0 | 12.76/106.22 | UNSAFE_SUPPORT |
| 5B4_LEVEL1_SAC | 16 | 0.35 | 0.59% | 0.29% | 0.10% | 0.769 | 59.3 | 0 | 1602.25/13616.06 | UNSAFE_SUPPORT |
| 5B4_LEVEL1_SAC | 16 | 0.50 | 0.10% | 0.00% | 0.00% | 0.705 | 60.4 | 0 | 4996.89/24776.51 | UNSAFE_SUPPORT |

## 15. Feasibility failures and catastrophic outliers

The per-cell feasibility failure breakdown is:

| Center | K | Scale | Displacement failure | Speed failure | Accel failure | Non-finite |
|---|---:|---:|---:|---:|---:|---:|
| ZERO | 4 | 0.05 | 11.52% | 0.00% | 0.00% | 0.00% |
| ZERO | 4 | 0.10 | 67.87% | 0.00% | 0.00% | 0.00% |
| ZERO | 4 | 0.20 | 94.73% | 10.35% | 0.00% | 0.00% |
| ZERO | 4 | 0.35 | 99.61% | 65.72% | 0.00% | 0.00% |
| ZERO | 4 | 0.50 | 99.90% | 90.53% | 0.00% | 0.00% |
| ZERO | 8 | 0.05 | 11.33% | 0.00% | 0.00% | 0.00% |
| ZERO | 8 | 0.10 | 68.85% | 0.00% | 0.00% | 0.00% |
| ZERO | 8 | 0.20 | 94.53% | 15.04% | 0.00% | 0.00% |
| ZERO | 8 | 0.35 | 99.41% | 78.61% | 0.00% | 0.00% |
| ZERO | 8 | 0.50 | 100.00% | 96.88% | 0.00% | 0.00% |
| ZERO | 16 | 0.05 | 11.33% | 0.00% | 0.00% | 0.00% |
| ZERO | 16 | 0.10 | 66.89% | 0.00% | 0.00% | 0.00% |
| ZERO | 16 | 0.20 | 94.82% | 15.14% | 0.00% | 0.00% |
| ZERO | 16 | 0.35 | 99.61% | 76.76% | 0.00% | 0.00% |
| ZERO | 16 | 0.50 | 99.90% | 96.19% | 0.00% | 0.00% |
| 5B4_LEVEL1_SAC | 4 | 0.05 | 48.93% | 0.00% | 0.00% | 0.00% |
| 5B4_LEVEL1_SAC | 4 | 0.10 | 77.34% | 0.00% | 0.00% | 0.00% |
| 5B4_LEVEL1_SAC | 4 | 0.20 | 94.34% | 11.91% | 0.00% | 0.00% |
| 5B4_LEVEL1_SAC | 4 | 0.35 | 99.80% | 65.92% | 0.00% | 0.00% |
| 5B4_LEVEL1_SAC | 4 | 0.50 | 100.00% | 91.60% | 0.00% | 0.00% |
| 5B4_LEVEL1_SAC | 8 | 0.05 | 50.10% | 0.00% | 0.00% | 0.00% |
| 5B4_LEVEL1_SAC | 8 | 0.10 | 76.95% | 0.00% | 0.00% | 0.00% |
| 5B4_LEVEL1_SAC | 8 | 0.20 | 95.61% | 15.82% | 0.00% | 0.00% |
| 5B4_LEVEL1_SAC | 8 | 0.35 | 99.22% | 80.86% | 0.00% | 0.00% |
| 5B4_LEVEL1_SAC | 8 | 0.50 | 99.90% | 97.27% | 0.00% | 0.00% |
| 5B4_LEVEL1_SAC | 16 | 0.05 | 49.22% | 0.00% | 0.00% | 0.00% |
| 5B4_LEVEL1_SAC | 16 | 0.10 | 78.61% | 0.00% | 0.00% | 0.00% |
| 5B4_LEVEL1_SAC | 16 | 0.20 | 94.14% | 17.58% | 0.00% | 0.00% |
| 5B4_LEVEL1_SAC | 16 | 0.35 | 99.41% | 79.59% | 0.00% | 0.00% |
| 5B4_LEVEL1_SAC | 16 | 0.50 | 99.90% | 97.07% | 0.00% | 0.00% |

Heavy-tail p99/max statistics are:

| Center | K | Scale | UAV speed p99/max (m/s) | UAV displacement p99/max (m) | Tip speed p99/max (m/s) |
|---|---:|---:|---:|---:|---:|
| ZERO | 4 | 0.05 | 1.01/1.21 | 0.66/0.88 | 1.97/2.37 |
| ZERO | 4 | 0.10 | 2.04/2.42 | 1.40/1.65 | 3.71/4.15 |
| ZERO | 4 | 0.20 | 4.07/4.69 | 2.64/3.13 | 7.16/8.42 |
| ZERO | 4 | 0.35 | 204.57/1318.70 | 27.80/122.49 | 206.45/1385.87 |
| ZERO | 4 | 0.50 | 1829.76/49701.19 | 248.94/4886.55 | 1856.69/50906.48 |
| ZERO | 8 | 0.05 | 1.10/1.32 | 0.71/0.78 | 2.13/2.43 |
| ZERO | 8 | 0.10 | 2.12/2.44 | 1.31/1.72 | 3.94/4.50 |
| ZERO | 8 | 0.20 | 4.74/79.32 | 2.89/6.56 | 8.43/80.43 |
| ZERO | 8 | 0.35 | 954.56/6399.77 | 91.11/593.71 | 991.26/6363.48 |
| ZERO | 8 | 0.50 | 4459.42/52161.42 | 601.27/5033.70 | 4553.29/53474.88 |
| ZERO | 16 | 0.05 | 1.05/1.30 | 0.66/0.87 | 2.11/2.53 |
| ZERO | 16 | 0.10 | 2.10/2.71 | 1.31/1.62 | 4.11/4.91 |
| ZERO | 16 | 0.20 | 5.06/78.80 | 2.82/9.96 | 8.79/80.66 |
| ZERO | 16 | 0.35 | 1508.88/5865.88 | 154.22/590.01 | 1619.76/5985.46 |
| ZERO | 16 | 0.50 | 11913.95/103596.42 | 1155.65/10779.80 | 12091.14/104746.20 |
| 5B4_LEVEL1_SAC | 4 | 0.05 | 1.31/1.54 | 0.94/1.12 | 2.61/3.01 |
| 5B4_LEVEL1_SAC | 4 | 0.10 | 2.27/2.54 | 1.56/1.85 | 4.28/5.29 |
| 5B4_LEVEL1_SAC | 4 | 0.20 | 4.08/6.43 | 2.72/3.49 | 7.44/14.16 |
| 5B4_LEVEL1_SAC | 4 | 0.35 | 238.55/1877.52 | 26.78/214.26 | 239.45/1935.56 |
| 5B4_LEVEL1_SAC | 4 | 0.50 | 1820.96/55067.47 | 191.80/7320.13 | 1837.77/57541.09 |
| 5B4_LEVEL1_SAC | 8 | 0.05 | 1.30/1.56 | 0.93/1.19 | 2.75/3.45 |
| 5B4_LEVEL1_SAC | 8 | 0.10 | 2.20/2.67 | 1.59/1.85 | 4.64/5.54 |
| 5B4_LEVEL1_SAC | 8 | 0.20 | 4.53/77.86 | 2.79/8.15 | 9.23/77.24 |
| 5B4_LEVEL1_SAC | 8 | 0.35 | 800.07/7458.08 | 90.72/776.01 | 816.22/8219.07 |
| 5B4_LEVEL1_SAC | 8 | 0.50 | 7130.40/52176.97 | 1026.67/7030.54 | 7162.60/53078.88 |
| 5B4_LEVEL1_SAC | 16 | 0.05 | 1.28/1.49 | 0.94/1.17 | 2.62/2.89 |
| 5B4_LEVEL1_SAC | 16 | 0.10 | 2.32/2.80 | 1.59/1.94 | 4.47/5.87 |
| 5B4_LEVEL1_SAC | 16 | 0.20 | 12.76/106.22 | 3.01/11.62 | 13.73/105.98 |
| 5B4_LEVEL1_SAC | 16 | 0.35 | 1602.25/13616.06 | 166.78/1282.18 | 1563.93/14687.27 |
| 5B4_LEVEL1_SAC | 16 | 0.50 | 4996.89/24776.51 | 517.87/4088.20 | 5150.27/25160.59 |

Displacement is the first support-limiting feasibility gate in low-noise cells; speed and catastrophic heavy tails emerge as scale increases. Command-acceleration and nonfinite failure rates remain zero because the radial decoder enforces the physical knot bound and every audited rollout remained finite.

## 16. Joint feasible-progress support and best region

Best region: center **5B4_LEVEL1_SAC**, K=4, scale=0.05. Feasible rate 51.07%; feasible & progress>=0.25 19.14%; feasible & progress>=0.50 0.00%; classification **USEFUL_SUPPORT**. Scientific successes: 0 in the best cell and 0 across the audit.

## 17. Center-A versus Center-B interpretation

Both centers have useful low-bandwidth/amplitude support; 5B.4's learned distribution was substantially too broad.

## 18. CEM reference only

CEM was not used as a center, covariance, scale, teacher, or dataset. Its known successful trajectory remains feasibility context only.

## 19. Recommended next methodological decision (not implemented)

Review a future actor-distribution initialization using the diagnosed bandwidth/amplitude region; do not train it yet.

## 20. Runtime, protection, and hardware

Runtime 321.118 s; grid rollouts 30720; peak CUDA memory 516.2 MiB. Protected test: **NOT EVALUATED**. Real hardware: **NOT EXECUTED**.

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Learning performed:
        NO

    Reward:
        rl_whip_reward_v2
        UNCHANGED

    Canonical context:
        FIXED

    Duration:
        1.20 s DIAGNOSTIC

    Catastrophic rollout reproduced:
        YES

    Command generation:
        HEALTHY

    Residual OOD observed:
        YES

    Residual-off removes runaway:
        NO

    Root-cause classification:
        CLOSED_LOOP_TRACKING_RUNAWAY

    Exploration centers:
        ZERO
        5B4_LEVEL1_SAC

    Bandwidths:
        K = 4, 8, 16

    Noise scales:
        0.05, 0.10, 0.20, 0.35, 0.50

    Best support region:
        center = 5B4_LEVEL1_SAC
        K = 4
        scale = 0.05

    Feasible rate:
        51.07 %

    Feasible & progress>=0.25:
        19.14 %

    Feasible & progress>=0.50:
        0.00 %

    Scientific successes in audit:
        0

    Support classification:
        USEFUL_SUPPORT

    Production model modified:
        NO

    SAC trained:
        NO

    CEM training data:
        NOT USED

    Physics conditioning:
        NOT ENABLED

    Protected test:
        NOT EVALUATED

    Real hardware:
        NOT EXECUTED
