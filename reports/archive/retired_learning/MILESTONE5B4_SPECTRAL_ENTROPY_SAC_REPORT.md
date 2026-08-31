# Milestone 5B.4 — Spectral Entropy SAC Report

## 1. Diagnosis from Milestone 5B.3

5B.3 showed healthy simulator, interface, reward-v2, critic conditioning, and entropy, but independent knot-space exploration failed to consolidate a whip. Stochastic samples occasionally approached the target while remaining unsafe or dynamically incorrect; the deterministic actor returned to a short no-op. This milestone tests temporal exploration geometry rather than another reward.

## 2. Frozen scientific contract

The simulator used `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI` unchanged. `rl_whip_reward_v2` is unchanged, as are every hard success gate. The task context is the single canonical settled state, target [1,0,1.4] m, direction [+1,0,0], and nominal theta. CEM data, demonstrations, physics randomization, target randomization, and initial-state randomization were not used.

## 3. Stationary reward and actor-only exploration pressure

The environment and replay store only `rl_whip_reward_v2`. The critic target remains the terminal environment reward with no bootstrap. `alpha_explore` appears only in the actor objective: `(alpha_SAC + alpha_explore) log pi - min(Q1,Q2)`. It never enters reward, replay, Q targets, or scientific metrics. The stationarity test used identical rewards at k=0, 100k, and 400k and passed.

## 4. Fixed duration curriculum

Duration was fixed at 1.20 s to remove the repeatedly observed 0.45-s escape route. This is a discovery curriculum, not a new scientific deadline or final policy formulation. The known 1.1175-s CEM solution lies within this rollout.

## 5. Old and new exploration formulations

The old actor sampled 49 approximately factorized raw knot/duration coordinates. The new actor samples 48 full-rank spectral coordinates: 16 coefficients independently for x, y, and z. All 16 frequencies are retained. Duration is deterministic, while the critic continues receiving a 49-D decoded action with normalized duration +1.

## 6. Exact orthonormal DCT convention and log probability

For temporal index n and frequency f, the inverse basis is `B[n,0]=sqrt(1/16)` and `B[n,f]=sqrt(2/16) cos(pi (n+1/2) f /16)` for f>0. Raw temporal values are `z=Bc`. The measured maximum `B^T B-I` error was 2.015e-15; rank was 16; reconstruction error was 1.332e-14; energy error was 5.684e-14. Since `|det B|=1`, the IDCT contributes zero log-Jacobian. Log probability is `log N(c;mu,sigma) - sum_k log|det J_radial(z_k)|`; fixed duration contributes zero.

## 7. Initial spectral standard deviation

The initial learned standard deviations were initialized to `1/sqrt(1+(f/4)^2)` for f=0..15 on each axis. All frequencies were nonzero; low frequencies were largest, and every log standard deviation remained trainable.

## 8. Distribution verification

Verification status: **PASS**. Checks: `{"all_frequency_std_nonzero": true, "analytic_log_probability_gradients_finite": true, "critic_target_stationary": true, "energy_preservation": true, "full_rank": true, "high_frequency_std_nonzero": true, "invertibility": true, "orthonormality": true, "radial_bound": true, "sample_log_probability_finite": true, "schedule_exact": true, "schedule_monotone_nonnegative": true}`.

## 9. Pre-training old-vs-new production exploration audit

Both distributions evaluated 2,048 actions with duration fixed at 1.20 s; no audit sample entered replay.

| Metric | Old direct-knot | New spectral |
|---|---:|---:|
| Mean temporal roughness | 417.299 | 181.190 |
| Roughness / 400 | 1.0432 | 0.4530 |
| Mean progress | 0.2872 | 0.2757 |
| 95th-percentile progress | 0.4652 | 0.3965 |
| Maximum progress | 0.9254 | 0.8162 |
| Feasible rate | 0.00% | 0.00% |
| d_min <= 0.20 m | 0.15% | 0.00% |
| Finite rate | 100.00% | 100.00% |

New/old mean roughness ratio: 0.4342. New raw spectral energy fractions: `{"high": 0.17467598617076874, "low": 0.5684564113616943, "mid": 0.25686758756637573}`. Low frequency energy dominated while high-frequency energy remained nonzero. Exploration-audit status: **PASS**.

## 10. Exact exploration schedule

`alpha_explore=0.25` through 75k episodes, decays linearly to zero over 75k–300k, and is zero thereafter. `alpha_SAC` remains independently learned against target entropy -48; `alpha_effective=alpha_SAC+alpha_explore` only in the actor loss.

## 11. SAC configuration

Fresh seed-42 actor, critics, and alpha; 83-D context; 48 stochastic spectral dimensions; critic external action 49; 256-256-256 SiLU networks; Adam 3e-4; Huber delta 1; 2,048 collection rows; 4,096 replay minibatch; 10,240 warmup; eight updates per batch; replay capacity 1,000,000; gradient clip 10; target entropy -48. No earlier SAC weights were resumed.

## 12. Pre-run learning checks

Status: **PASS**. `{"actor_critic_backward_finite": true, "cem_training_data_used": false, "checkpoint_save_resume": true, "critic_target_reward_only": true, "duration_normalized_fixed_at_one": true, "external_action_shape_49": true, "fixed_canonical_context_identical": true}`

## 13. Deterministic policy progression

| Episodes | Level | Progress | d_min (mm) | Directed (m/s) | Direction (deg) | Feasible | Success | alpha_SAC | alpha_explore | Stochastic successes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0053 | 1343.937 | 0.010 | 39.618 | True | False | 1.0000 | 0.2500 | 0 |
| 10240 | 0 | 0.0072 | 1341.354 | 0.098 | 16.758 | True | False | 0.9976 | 0.2500 | 0 |
| 20480 | 0 | 0.0431 | 1292.874 | 0.698 | 29.086 | True | False | 0.9854 | 0.2500 | 0 |
| 30720 | 0 | 0.0441 | 1291.530 | 0.169 | 49.305 | True | False | 0.9731 | 0.2500 | 0 |
| 40960 | 0 | 0.1574 | 1138.404 | 0.251 | 44.788 | True | False | 0.9611 | 0.2500 | 0 |
| 51200 | 0 | 0.0449 | 1290.473 | 0.256 | 44.441 | True | False | 0.9494 | 0.2500 | 0 |
| 61440 | 0 | 0.1165 | 1193.689 | 0.312 | 33.322 | True | False | 0.9379 | 0.2500 | 0 |
| 71680 | 0 | 0.1154 | 1195.228 | 0.184 | 56.090 | True | False | 0.9266 | 0.2500 | 0 |
| 81920 | 0 | 0.0520 | 1280.793 | 0.222 | 55.331 | True | False | 0.9155 | 0.2423 | 0 |
| 90112 | 0 | 0.0687 | 1258.332 | 0.298 | 46.816 | True | False | 0.9066 | 0.2332 | 0 |
| 100352 | 0 | 0.1275 | 1178.862 | 0.446 | 22.206 | True | False | 0.8957 | 0.2218 | 0 |
| 110592 | 0 | 0.1093 | 1203.359 | 0.460 | 19.784 | True | False | 0.8850 | 0.2105 | 0 |
| 120832 | 0 | 0.1504 | 1147.875 | 0.334 | 45.960 | True | False | 0.8744 | 0.1991 | 0 |
| 131072 | 0 | 0.1035 | 1211.226 | 0.215 | 45.690 | True | False | 0.8639 | 0.1877 | 0 |
| 141312 | 0 | 0.1236 | 1184.139 | 0.291 | 53.152 | True | False | 0.8535 | 0.1763 | 0 |
| 151552 | 0 | 0.1041 | 1210.407 | 0.279 | 33.654 | True | False | 0.8433 | 0.1649 | 0 |
| 161792 | 0 | 0.1000 | 1216.043 | 0.269 | 54.989 | True | False | 0.8332 | 0.1536 | 0 |
| 172032 | 0 | 0.0819 | 1240.438 | 0.363 | 23.814 | True | False | 0.8232 | 0.1422 | 0 |
| 180224 | 0 | 0.1250 | 1182.167 | 0.205 | 45.950 | True | False | 0.8153 | 0.1331 | 0 |
| 190464 | 0 | 0.0991 | 1217.145 | 0.217 | 52.437 | True | False | 0.8056 | 0.1217 | 0 |
| 200704 | 0 | 0.1782 | 1110.299 | 0.294 | 50.310 | True | False | 0.7959 | 0.1103 | 0 |
| 210944 | 0 | 0.1187 | 1190.739 | 0.343 | 49.875 | True | False | 0.7864 | 0.0990 | 0 |
| 221184 | 0 | 0.1455 | 1154.516 | 0.348 | 44.057 | True | False | 0.7770 | 0.0876 | 0 |
| 231424 | 0 | 0.0739 | 1251.272 | 0.234 | 33.587 | True | False | 0.7677 | 0.0762 | 0 |
| 241664 | 0 | 0.0855 | 1235.537 | 0.311 | 41.237 | True | False | 0.7585 | 0.0648 | 0 |
| 251904 | 0 | 0.1409 | 1160.782 | 0.286 | 61.707 | True | False | 0.7495 | 0.0534 | 0 |
| 260096 | 0 | 0.1310 | 1174.147 | 0.279 | 54.825 | True | False | 0.7423 | 0.0443 | 0 |
| 270336 | 0 | 0.1562 | 1140.043 | 0.329 | 56.398 | True | False | 0.7334 | 0.0330 | 0 |
| 280576 | 0 | 0.0925 | 1226.179 | 0.371 | 54.862 | True | False | 0.7247 | 0.0216 | 0 |
| 290816 | 0 | 0.1449 | 1155.360 | 0.466 | 39.785 | True | False | 0.7160 | 0.0102 | 0 |
| 301056 | 0 | 0.1657 | 1127.240 | 0.483 | 40.054 | True | False | 0.7075 | 0.0000 | 0 |
| 311296 | 0 | 0.1721 | 1118.553 | 0.386 | 57.671 | True | False | 0.6990 | 0.0000 | 0 |
| 321536 | 0 | 0.1836 | 1103.076 | 0.518 | 47.700 | True | False | 0.6907 | 0.0000 | 0 |
| 331776 | 0 | 0.1437 | 1156.982 | 0.322 | 49.913 | True | False | 0.6824 | 0.0000 | 0 |
| 342016 | 0 | 0.2229 | 1049.907 | 0.505 | 33.748 | True | False | 0.6743 | 0.0000 | 0 |
| 350208 | 0 | 0.1905 | 1093.758 | 0.380 | 49.451 | True | False | 0.6678 | 0.0000 | 0 |
| 360448 | 0 | 0.1922 | 1091.421 | 0.583 | 33.496 | True | False | 0.6598 | 0.0000 | 0 |
| 370688 | 0 | 0.2365 | 1031.615 | 0.795 | 30.539 | True | False | 0.6520 | 0.0000 | 0 |
| 380928 | 0 | 0.2372 | 1030.674 | 0.697 | 24.326 | True | False | 0.6442 | 0.0000 | 0 |
| 391168 | 0 | 0.1276 | 1178.630 | 0.517 | 20.415 | True | False | 0.6365 | 0.0000 | 0 |
| 401408 | 0 | 0.1886 | 1096.242 | 0.445 | 34.243 | True | False | 0.6289 | 0.0000 | 0 |
| 411648 | 0 | 0.1754 | 1114.058 | 0.459 | 55.718 | True | False | 0.6214 | 0.0000 | 0 |
| 421888 | 0 | 0.1994 | 1081.639 | 0.385 | 50.719 | True | False | 0.6140 | 0.0000 | 0 |
| 430080 | 0 | 0.1587 | 1136.616 | 0.420 | 49.482 | True | False | 0.6081 | 0.0000 | 0 |
| 440320 | 0 | 0.1943 | 1088.520 | 0.355 | 57.174 | True | False | 0.6008 | 0.0000 | 0 |
| 450560 | 0 | 0.2367 | 1031.291 | 0.408 | 60.859 | True | False | 0.5937 | 0.0000 | 0 |
| 460800 | 0 | 0.1925 | 1090.966 | 0.419 | 57.326 | True | False | 0.5866 | 0.0000 | 0 |
| 471040 | 0 | 0.2380 | 1029.577 | 0.562 | 32.617 | True | False | 0.5796 | 0.0000 | 0 |
| 481280 | 1 | 0.2549 | 1006.757 | 0.675 | 37.558 | True | False | 0.5727 | 0.0000 | 0 |
| 491520 | 1 | 0.2987 | 947.468 | 0.825 | 36.697 | True | False | 0.5658 | 0.0000 | 0 |
| 501760 | 0 | 0.1667 | 1125.852 | 0.434 | 52.408 | True | False | 0.5591 | 0.0000 | 0 |

Highest deterministic behavior level: **1**.

## 14. Critic diagnostics

Q1 Huber loss range [0.154511, 14.280214], Q2 range [0.154336, 14.317763]. Reward/Q correlation moved from 0.0013 to 0.2682. Replay environment rewards remained in [-16.1106, 3.6710].
The report artifacts include Q/reward correlations, Q ranges, replay reward ranges, and critic gradient norms at each evaluation. No NaN/Inf was accepted.

## 15. Actor, entropy, and spectral diagnostics

Policy entropy during updates ranged from 12.1111 to 22.3288.
alpha_SAC moved from 0.9976 to 0.5591; alpha_effective began at 1.2476 and ended at 0.5591. Initial low/mid/high sampled spectral energy fractions were 0.570/0.257/0.172; final fractions were 0.245/0.252/0.503. Initial low/mid/high actor standard deviations were 0.916/0.596/0.339; final values were 0.735/0.735/0.739.

`training_history.json` records alpha_SAC, alpha_explore, alpha_effective, entropy, mean log_pi, actor loss, acceleration norms, roughness, sign changes, and stochastic behavior. `spectral_statistics_history.json` preserves the complete per-collection spectral evolution.

## 16. Stochastic progress, feasibility, and behavioral diversity

Across collection batches, maximum sampled progress reached 0.993058, maximum batch p95 progress was 0.452532, and minimum sampled d_min was 9.380 mm. Maximum reported directed speed was 31.290 m/s. However, stochastic feasible rate was 0.00%–0.00%; no feasible high-progress joint event occurred. The maxima of interval fractions were: progress>=0.25 71.68%, >=0.50 3.52%, >=0.75 0.63%, >=0.90 0.15%, d_min<=0.20 m 0.34%, and d_min<=0.10 m 0.10%. Every feasible-and-progress/distance joint fraction was 0%.

The stochastic population remained tensor-finite, but unsafe outliers were extreme: maximum tip speed 562927.4 m/s, UAV displacement 52532.4 m, and UAV speed 562529.1 m/s. These rows were correctly infeasible and received finite bounded reward penalties; they also dominated naive descriptor dispersion. Mean normalized descriptor distance changed from 1925.928 to 1385.291. No novelty term was added.

## 17. Stochastic scientific-success diagnostics

First stochastic success: **NONE**. Total stochastic successes: **0**. Best stochastic success metadata: `NONE`.
Successful samples, if any, remained ordinary replay entries and were neither duplicated nor treated as expert data.

## 18. Selected checkpoint and authoritative replay

Selection used deterministic success, behavior level, feasibility, progress, distance, then speed/direction. Selected deterministic level: 1; progress: 0.298739; reward: 1.194818. The final action was replayed once at batch size one from the canonical state with no noise and fixed T=1.20 s.

| Scientific gate | Authoritative result | Pass |
|---|---:|---:|
| Tip distance <= 50 mm | 948.482 mm | False |
| Directed speed >= 4 m/s | 0.835 m/s | False |
| Direction <= 30 deg | 34.665 deg | False |
| c10 first | None | False |
| UAV displacement <= 0.50 m | 0.4082 m | True |
| UAV speed <= 3 m/s | 0.4910 m/s | True |
| Acceleration <= 20 m/s² | 2.8359 m/s² | True |
| Finite rollout | True | True |

## 19. CEM feasibility reference only

The existing CEM reference (not used for training) remains: 1.756-mm tip error, 4.599-m/s directed speed, 19.615-deg direction error, 1.1175-s duration, scientific PASS. It establishes that the simulator/action task is feasible.

## 20. Runtime and artifacts

Collected episodes: 501760; gradient updates: 1928; training runtime: 435.484 s; overall runtime: 454.182 s; peak CUDA memory: 95.8 MiB. Video: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\policy_training\sac_canonical_spectral_entropy_v1\2026-08-30T015034.715379Z\oneshot_sac_spectral_entropy_canonical_final_replay.mp4`.

## 21. Scientific interpretation and classification

**SAC_STRUCTURED_EXPLORATION_LIMITED**. No scientific stochastic success appeared and the deterministic actor did not exceed Level 2.

Protected test: **NOT EVALUATED**. Real hardware: **NOT EXECUTED**. Every command remains simulation-only and not authorized for real flight.

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Policy:
        One-Shot Terminal SAC

    Context:
        SINGLE CANONICAL

    Execution:
        OPEN LOOP

    Reward:
        rl_whip_reward_v2
        UNCHANGED

    Exploration representation:
        FULL-RANK ORTHONORMAL DCT

    Acceleration knots:
        16 x 3

    Stochastic dimensions:
        48

    Duration:
        FIXED 1.20 s
        DISCOVERY CURRICULUM ONLY

    Base target entropy:
        -48

    Early exploration bonus:
        alpha_explore initial = 0.25

    Exploration anneal:
        hold through 75k
        linear -> 0 by 300k

    CEM training data:
        NOT USED

    Episodes:
        501760

    First stochastic success:
        episode NONE

    Total stochastic successes:
        0

    Best stochastic success:
        NONE

    Highest deterministic behavior level:
        1

    Deterministic progress:
        0.297989

    Deterministic minimum tip distance:
        948.482 mm

    Deterministic directed speed:
        0.835 m/s

    Deterministic direction error:
        34.665 deg

    Deterministic UAV displacement:
        0.4082 m

    Deterministic feasible:
        YES

    Deterministic scientific task:
        FAIL

    Result:
        SAC_STRUCTURED_EXPLORATION_LIMITED

    Physics conditioning:
        NOT ENABLED

    Protected test:
        NOT EVALUATED

    Real hardware:
        NOT EXECUTED
