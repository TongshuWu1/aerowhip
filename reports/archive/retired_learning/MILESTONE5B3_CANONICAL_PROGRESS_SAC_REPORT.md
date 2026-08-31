# Milestone 5B.3 — Progress-Shaped RL Reward and Canonical SAC Discovery Pilot

## 1. Diagnosis from Milestone 5B.2

Milestone 5B.2 fixed reward and critic numerical conditioning, but the 0.20-m proximity kernel was effectively zero over most untrained behavior. Safe no-op therefore returned approximately zero while aggressive exploration often paid safety/control penalties. The actor converged toward short no-whip maneuvers.

## 2. Exact reward-version difference

`rl_whip_reward_v1` remains preserved. `rl_whip_reward_v2` changes exactly one term: `R_progress = 4 * (d0-d_min)/max(d0,eps)`, clamped by construction to [0,4]. Every strike, success, safety, non-tip, control, and successful-only time term is unchanged.

## 3. Reward audit

| Trajectory | Success | Feasible | d0 [m] | d_min [m] | Progress | R_progress | R_v1 | R_v2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strong_success | True | True | 1.3511 | 0.0018 | 0.9987 | 3.9948 | 9.4597 | 13.4545 |
| fragile_success | True | True | 1.3511 | 0.0011 | 0.9992 | 3.9967 | 9.3748 | 13.3715 |
| close_wrong_direction | False | True | 1.3511 | 0.0055 | 0.9959 | 3.9838 | 4.3877 | 8.3715 |
| close_slow | False | True | 1.3511 | 0.0117 | 0.9914 | 3.9655 | 3.7700 | 7.7355 |
| old_aggressive_mppi | False | False | 1.3511 | 0.0564 | 0.9582 | 3.8330 | 1.5879 | 5.4208 |
| failed_sac | False | False | 1.3511 | 1.0825 | 0.1988 | 0.7951 | -5.3056 | -4.5106 |
| safe_hover_no_whip | False | True | 1.3511 | 1.3445 | 0.0049 | 0.0195 | 0.0000 | 0.0195 |
| bad_random_00 | False | False | 1.3511 | 0.9240 | 0.3161 | 1.2646 | -16.1233 | -14.8587 |

Observed reward-v2 range: [-15.584900, 13.454471]. The maximum numerical error in `v2 = v1 + 4*progress` was 1.776e-15.

## 4. Random progress bins

| Progress bin | Count | Feasible | Mean reward | Median reward | Feasible mean | Feasible median |
|---|---:|---:|---:|---:|---:|---:|
| [0.00, 0.10) | 1 | 1 | -0.1026 | -0.1026 | -0.1026 | -0.1026 |
| [0.10, 0.25) | 48 | 1 | -11.1048 | -15.1638 | 0.5869 | 0.5869 |
| [0.25, 0.50) | 77 | 0 | -13.0863 | -14.8843 | — | — |
| [0.50, 0.75) | 2 | 0 | -13.9314 | -13.9314 | — | — |
| [0.75, 1.00) | 0 | 0 | — | — | — | — |

**REWARD_AUDIT: PASS**

CEM artifacts were used only as saved reward-audit references. CEM actions, elites, trajectories, and demonstrations were not inserted into replay and did not initialize the actor.

## 5. Canonical-only SAC setup

Every training and validation episode used the exact same canonical settled state, target [1,0,1.4] m, direction [+1,0,0], and nominal theta. The existing fixed 5B.2 normalizer was reused. Only stochastic actions and resulting rewards varied.

The actor/twin-critic 256-256-256 SiLU architecture, radial action transform, terminal reward-only target, automatic alpha, Huber delta 1, 2,048 collection batch, 4,096 replay minibatch, eight updates per batch, and learning rates 3e-4 were unchanged.

## 6. Learning progression

| Episodes | Level | Progress | d_min [mm] | Directed [m/s] | Direction [deg] | Duration [s] | Feasible | Success | Reward | Q/reward corr. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0025 | 1347.748 | 0.010 | 39.884 | 0.8250 | True | False | 0.0099 | — |
| 10240 | 0 | 0.0078 | 1340.619 | 0.033 | 38.100 | 0.8260 | True | False | 0.0310 | -0.0426 |
| 20480 | 0 | 0.1803 | 1107.462 | 1.266 | 52.399 | 0.7550 | True | False | 0.7195 | 0.3084 |
| 30720 | 1 | 0.2877 | 962.364 | 1.468 | 47.723 | 0.6633 | False | False | 1.0527 | 0.4784 |
| 40960 | 1 | 0.2682 | 988.748 | 0.017 | 32.732 | 0.5725 | False | False | 0.4976 | 0.5068 |
| 51200 | 1 | 0.2506 | 1012.552 | 0.022 | 40.143 | 0.5178 | False | False | 0.1988 | 0.5420 |
| 61440 | 0 | 0.1713 | 1119.700 | 0.786 | 67.258 | 0.4866 | True | False | 0.6800 | 0.6022 |
| 71680 | 0 | 0.0000 | 1351.367 | 0.011 | 86.434 | 0.4727 | True | False | -0.0015 | 0.5733 |
| 81920 | 0 | 0.0000 | 1351.319 | 0.004 | 88.462 | 0.4674 | True | False | -0.0014 | 0.5942 |
| 90112 | 0 | 0.0049 | 1344.432 | 0.241 | 58.424 | 0.4637 | True | False | 0.0189 | 0.6057 |
| 100352 | 0 | 0.0070 | 1341.659 | 0.172 | 65.997 | 0.4610 | True | False | 0.0272 | 0.6212 |
| 110592 | 0 | 0.0272 | 1314.388 | 0.213 | 67.168 | 0.4607 | True | False | 0.1080 | 0.6326 |
| 120832 | 0 | 0.0384 | 1299.273 | 0.196 | 73.607 | 0.4600 | True | False | 0.1527 | 0.6419 |
| 131072 | 0 | 0.0338 | 1305.382 | 0.215 | 71.955 | 0.4605 | True | False | 0.1348 | 0.6487 |
| 141312 | 0 | 0.0439 | 1291.794 | 0.193 | 74.301 | 0.4605 | True | False | 0.1749 | 0.6564 |
| 151552 | 0 | 0.0401 | 1296.958 | 0.203 | 75.346 | 0.4609 | True | False | 0.1596 | 0.6637 |
| 161792 | 0 | 0.0520 | 1280.864 | 0.224 | 72.381 | 0.4610 | True | False | 0.2072 | 0.6704 |
| 172032 | 0 | 0.0501 | 1283.367 | 0.222 | 73.695 | 0.4604 | True | False | 0.1998 | 0.6746 |
| 180224 | 0 | 0.0574 | 1273.540 | 0.254 | 72.296 | 0.4605 | True | False | 0.2288 | 0.6778 |
| 190464 | 0 | 0.0461 | 1288.760 | 0.190 | 76.403 | 0.4611 | True | False | 0.1838 | 0.6822 |
| 200704 | 0 | 0.0473 | 1287.251 | 0.000 | 89.933 | 0.4606 | True | False | 0.1882 | 0.6833 |
| 210944 | 0 | 0.0525 | 1280.182 | 0.002 | 87.236 | 0.4611 | True | False | 0.2091 | 0.6845 |
| 221184 | 0 | 0.0513 | 1281.826 | 0.200 | 76.455 | 0.4613 | True | False | 0.2042 | 0.6861 |
| 231424 | 0 | 0.0547 | 1277.195 | 0.212 | 75.477 | 0.4611 | True | False | 0.2180 | 0.6861 |
| 241664 | 0 | 0.0584 | 1272.217 | 0.280 | 70.107 | 0.4613 | True | False | 0.2327 | 0.6864 |
| 251904 | 0 | 0.0523 | 1280.383 | 0.256 | 72.113 | 0.4615 | True | False | 0.2085 | 0.6863 |
| 260096 | 0 | 0.0442 | 1291.341 | 0.239 | 71.916 | 0.4613 | True | False | 0.1762 | 0.6853 |
| 270336 | 0 | 0.0496 | 1284.083 | 0.247 | 72.159 | 0.4614 | True | False | 0.1977 | 0.6864 |
| 280576 | 0 | 0.0458 | 1289.238 | 0.235 | 72.596 | 0.4617 | True | False | 0.1824 | 0.6854 |
| 290816 | 0 | 0.0567 | 1274.513 | 0.249 | 73.564 | 0.4607 | True | False | 0.2259 | 0.6879 |
| 301056 | 0 | 0.0528 | 1279.762 | 0.174 | 77.622 | 0.4607 | True | False | 0.2104 | 0.6870 |
| 311296 | 0 | 0.0551 | 1276.690 | 0.231 | 73.798 | 0.4613 | True | False | 0.2195 | 0.6862 |
| 321536 | 0 | 0.0570 | 1274.146 | 0.197 | 76.380 | 0.4608 | True | False | 0.2270 | 0.6853 |
| 331776 | 0 | 0.0549 | 1276.883 | 0.215 | 74.895 | 0.4613 | True | False | 0.2189 | 0.6843 |
| 342016 | 0 | 0.0535 | 1278.802 | 0.188 | 77.124 | 0.4615 | True | False | 0.2132 | 0.6832 |
| 350208 | 0 | 0.0560 | 1275.381 | 0.205 | 75.911 | 0.4607 | True | False | 0.2233 | 0.6842 |
| 360448 | 0 | 0.0540 | 1278.125 | 0.218 | 74.760 | 0.4610 | True | False | 0.2153 | 0.6854 |
| 370688 | 0 | 0.0542 | 1277.913 | 0.169 | 78.438 | 0.4609 | True | False | 0.2159 | 0.6830 |
| 380928 | 0 | 0.0589 | 1271.520 | 0.271 | 71.653 | 0.4603 | True | False | 0.2348 | 0.6832 |
| 391168 | 0 | 0.0540 | 1278.178 | 0.226 | 73.568 | 0.4607 | True | False | 0.2151 | 0.6833 |
| 401408 | 0 | 0.0551 | 1276.660 | 0.236 | 74.243 | 0.4611 | True | False | 0.2195 | 0.6831 |
| 411648 | 0 | 0.0557 | 1275.849 | 0.226 | 74.173 | 0.4612 | True | False | 0.2220 | 0.6824 |
| 421888 | 0 | 0.0493 | 1284.440 | 0.251 | 71.954 | 0.4609 | True | False | 0.1966 | 0.6829 |
| 430080 | 0 | 0.0520 | 1280.883 | 0.237 | 72.986 | 0.4607 | True | False | 0.2071 | 0.6821 |
| 440320 | 0 | 0.0512 | 1281.906 | 0.201 | 76.224 | 0.4601 | True | False | 0.2041 | 0.6808 |
| 450560 | 0 | 0.0557 | 1275.857 | 0.227 | 74.696 | 0.4605 | True | False | 0.2219 | 0.6829 |
| 460800 | 0 | 0.0570 | 1274.139 | 0.259 | 72.037 | 0.4603 | True | False | 0.2270 | 0.6816 |
| 471040 | 0 | 0.0498 | 1283.815 | 0.184 | 76.900 | 0.4602 | True | False | 0.1984 | 0.6819 |
| 481280 | 0 | 0.0590 | 1271.402 | 0.193 | 77.253 | 0.4603 | True | False | 0.2351 | 0.6826 |
| 491520 | 0 | 0.0493 | 1284.489 | 0.183 | 77.447 | 0.4599 | True | False | 0.1963 | 0.6822 |
| 501760 | 0 | 0.0492 | 1284.616 | 0.155 | 79.100 | 0.4593 | True | False | 0.1960 | 0.6835 |

Highest behavior level reached: **1**. Selected checkpoint level: **1**.

## 7. Critic and actor diagnostics

Critic Huber losses remained in [1.145488, 10.557715]

Reward-Q correlation progressed from -0.0426 to 0.6835, with a maximum of 0.6879. Actor loss remained in [-21.8644, -6.5159]. Alpha moved from 0.9987 to 0.5634; entropy remained in [19.7194, 22.9563] and ended at 19.7213. No NaN/Inf or critic explosion occurred.

Mean stochastic duration moved from 0.8307 s to 0.4658 s. Mean normalized acceleration-knot norm remained in [0.6966, 0.7800], and normalized action standard deviation remained in [0.4196, 0.4693]. Collection batches contained no successes; their maximum per-row progress reached 0.9885, but those high-progress samples did not satisfy the hard gates. Collection feasibility ranged from 1.51% to 35.06%.

Episodes: 501760; gradient updates: 1928; training runtime: 430.607 s. Duration diagnostic: **LOWER_DURATION_ATTRACTOR_PERSISTS**.

## 8. Selected deterministic checkpoint and authoritative replay

The fixed-2,048-row checkpoint evaluation reported reward 1.052735 and progress 0.287714. The authoritative batch-one replay reported reward 1.035142, progress 0.283317, and R_progress 1.133267. Both remain Level 1 and have the same hard-gate classification. Authoritative replay metrics are used below.

| Gate | Result | Pass |
|---|---:|---:|
| Tip distance <= 50 mm | 968.305 mm | False |
| Directed speed >= 4 m/s | 1.474 m/s | False |
| Direction <= 30 deg | 47.663 deg | False |
| c10 first | None | False |
| UAV displacement <= 0.50 m | 0.6099 m | False |
| UAV speed <= 3 m/s | 1.6720 m/s | True |
| Command acceleration <= 20 m/s² | 4.5552 m/s² | True |
| Finite rollout | True | True |

## 9. CEM reference only

The saved CEM reference remains: tip error 1.756 mm, directed speed 4.599 m/s, direction error 19.615 deg, and duration 1.1175 s. It proves feasibility but was not used for SAC training.

## 10. Classification and boundaries

**SAC_CANONICAL_EXPLORATION_LIMITED**

Video: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\policy_training\sac_canonical_progress_v1\2026-08-30T001833.216226Z\oneshot_sac_progress_canonical_final_replay.mp4`

Physics conditioning: **NOT ENABLED**. Protected test: **NOT EVALUATED**. Real hardware: **NOT EXECUTED**.

## 11. Verification

Reward-v2 identity/progress tests, fixed-context collection checks, finite actor/environment and Huber-update checks, and checkpoint save/resume passed. The focused learning suite passed 20 tests, the complete repository regression suite passed 90 tests in 38.92 s, and compileall passed.

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Policy:
        One-Shot Terminal SAC

    Context:
        SINGLE CANONICAL CONTEXT

    Execution:
        OPEN LOOP

    CEM training data:
        NOT USED

    Reward:
        rl_whip_reward_v2

    Progress weight:
        4.0

    Reward audit:
        PASS

    SAC episodes:
        501760

    Highest behavior level:
        1

    Progress:
        0.283317

    Minimum tip distance:
        968.305 mm

    Directed speed:
        1.474 m/s

    Direction error:
        47.663 deg

    Duration:
        0.663268 s

    Feasible:
        NO

    Scientific task:
        FAIL

    SAC result:
        SAC_CANONICAL_EXPLORATION_LIMITED

    Duration diagnostic:
        LOWER_DURATION_ATTRACTOR_PERSISTS

    Physics conditioning:
        NOT ENABLED

    Protected test:
        NOT EVALUATED

    Real hardware:
        NOT EXECUTED
