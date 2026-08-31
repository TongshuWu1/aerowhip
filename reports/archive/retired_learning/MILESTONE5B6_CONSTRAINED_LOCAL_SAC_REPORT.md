# Milestone 5B.6 — Feasibility-Constrained Local Spectral SAC

## 1. Scientific diagnosis from 5B.5

5B.5 identified CLOSED_LOOP_TRACKING_RUNAWAY: commands were internally consistent, but broad aggressive actions left the effective model's tracking domain and unsaturated Kp/Kv feedback amplified error. It also found useful local support around the SAC-derived Level-1 action at K=4, scale 0.05 (51.07% feasible; 19.14% feasible and progress>=0.25). The present experiment therefore used local constrained improvement rather than broader entropy.

## 2. Frozen model and immutable task

The model remained `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`: no controller, residual, cable, DDER, precision, or projection parameter changed. `rl_whip_reward_v2` weights and every scientific success gate remained unchanged. Theta was nominal only. Protected data and hardware were not accessed.

## 3. Hard-safety failure quarantine

The learning evaluator used `HARD_SAFETY_FAILURE_QUARANTINE`. A row was stopped at its first UAV-displacement, UAV-speed, command-acceleration, or non-finite safety violation. Task terms were accumulated only through the immediately preceding valid prefix (`RL_V2_HARD_SAFETY_PREFIX`); failed rows were forced unsuccessful. This is a validity-domain rule at the learning layer, not a simulator or reward-weight change.

Feasible-rollout equivalence: **PASS**. Maximum metric difference `0.000e+00` and maximum UAV/cable trajectory difference `0.000e+00` over the Level-1 center plus 32 feasible support samples.

Failed-prefix audit: **PASS**. Post-failure progress/near-target credit was excluded, no failed row became successful, and quarantined recorded speed stayed below the first 3-m/s boundary instead of propagating 1e3–1e5-m/s runaway dynamics.

## 4. SAC-derived local policy center

The exact 5B4 Level-1 SAC action was reconstructed via inverse radial squash/DCT from the 5B.5 artifact. Reconstruction status: **PASS**, maximum normalized-action error `3.446e-08`. No CEM action, covariance, elite, demonstration, or expert buffer was used.

## 5. K=4 local spectral formulation

The actor outputs a 12-D Gaussian offset for frequencies f=0..3 on x/y/z. Frequencies 4..15 remain exactly at the Level-1 center. IDCT, the existing radial squash, and the physical <=20-m/s^2 decoder are unchanged. The mean head was initialized to zero, exactly reproducing the center. Standard deviations were initialized at `0.05/sqrt(1+(f/4)^2)` and constrained throughout to `[0.005,0.10]` times that profile. Duration was fixed at 1.20 s for this discovery curriculum only.

## 6. Entropy calibration

The target entropy was calibrated from 100,000 initialized transformed-action samples: H_init = **-20.059059** (sample std 2.428536). It was frozen for training. Ordinary automatic-temperature SAC was used; `alpha_explore` was removed.

## 7. Constrained terminal SAC

Twin task critics regress the stationary valid-prefix `rl_whip_reward_v2` return with Huber delta 1. Twin safety critics regress binary terminal safety cost using BCEWithLogits. The actor objective is `alpha*log_pi - min(Q_R1,Q_R2) + lambda_safe*max(sigmoid(Q_C1),sigmoid(Q_C2))`. Lambda is nonnegative, initialized to 1, updated by dual ascent against failure target 0.20, and capped at 100 only for numerical protection. There is no bootstrap, next state, or simulator gradient.

## 8. Initial replay/support reproduction

Before any update, 20,480 local episodes produced feasible rate **49.37%**, feasible-and-progress>=0.25 **17.95%**, and feasible-and-progress>=0.50 **0.05%**. Gate: **PASS** (`{"feasible_progress_ge_0_50_remains_rare": true, "feasible_rate_reproduced": true, "useful_joint_support_reproduced": true}`).

## 9. Training configuration

Seed 42; Adam 3e-4 for actor, reward critics, safety critics, alpha, and lambda; collection batch 2,048; replay minibatch 4,096; 20,480 initial rows; eight updates per collection; replay capacity 1,000,000; gradient clip 10; maximum 500,000 episodes or 45 minutes. Context and target stayed exactly canonical, physics nominal, and execution one-query open loop.

## 10. Deterministic progression

| Episodes | Level | Progress | d_min (mm) | Directed (m/s) | Direction (deg) | Feasible | Success | p_fail | lambda |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.2980 | 948.50 | 0.947 | 35.96 | True | False | 0.509 | 1.000 |
| 20480 | 1 | 0.2980 | 948.50 | 0.947 | 35.96 | True | False | 0.509 | 1.000 |
| 30720 | 0 | 0.0599 | 1270.15 | 0.418 | 43.24 | True | False | 0.526 | 1.008 |
| 40960 | 0 | 0.0000 | 1351.19 | 0.848 | 45.11 | False | False | 0.496 | 1.016 |
| 51200 | 0 | 0.1776 | 1111.19 | 0.634 | 57.66 | False | False | 0.697 | 1.024 |
| 61440 | 2 | 0.5042 | 669.81 | 1.030 | 54.65 | True | False | 0.506 | 1.035 |
| 71680 | 0 | 0.1555 | 1141.01 | 0.011 | 86.10 | False | False | 0.994 | 1.043 |
| 81920 | 0 | 0.1268 | 1179.74 | 0.379 | 40.82 | True | False | 0.687 | 1.053 |

Highest deterministic behavior level: **2**.

## 11. Stochastic feasible-progress support

Final recorded collection: feasible **13.82%**; feasible-and-progress>=0.25 **1.81%**; >=0.50 **0.10%**; >=0.75 **0.00%**; scientific success rate **0.0000%**.

## 12. Safety-critic and constraint diagnostics

Final collection safety AUROC `0.1948968458142724`, Brier score `0.197328`, actual failure `86.18%`, predicted failure `60.94%`.
Final alpha `0.918892`; final lambda `1.052693`; target failure probability 0.20. Lambda-cap reached: **False**.

## 13. Reward-critic and actor/std diagnostics

Final fixed-sample reward/Q correlation `0.583996057510376`; replay reward range `[-0.0006575027946382761, 2.2241103649139404]`; Q1 range `[0.5578954219818115, 0.7294809222221375]`; Q2 range `[0.5541772246360779, 0.7282786965370178]`.
All saved update statistics were finite. Active K=4 standard deviations and the deterministic spectral-offset norm are recorded in `local_spectral_history.json`; bounds were enforced by construction.

## 14. Scientific successes and persisted actions

First stochastic scientific success: **NONE**. Total stochastic scientific successes: **0**. Exact normalized tensors and RNG provenance were persisted for the required best/representative categories in `persisted_key_actions.npz` and its manifest. They remain diagnostic replay records, not expert data.

## 15. Selected checkpoint and authoritative replay

Checkpoint selection prioritized deterministic success, level, feasibility, progress, distance, and speed/direction. The selected action was replayed once at batch size one through the **unmodified full production simulator**, not the quarantine path. Video: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\policy_training\constrained_sac_local_k4_v1\2026-08-30T054910.315475Z\oneshot_constrained_sac_k4_canonical_final_replay.mp4`.

| Scientific gate | Authoritative full-production result | Pass |
|---|---:|---:|
| Tip distance <= 50 mm | 658.778 mm | False |
| Directed speed >= 4 m/s | 1.017 m/s | False |
| Direction error <= 30 deg | 55.005 deg | False |
| c10 first | None | False |
| UAV displacement <= 0.50 m | 0.4290 m | True |
| UAV speed <= 3.0 m/s | 1.7440 m/s | True |
| Command acceleration <= 20 m/s^2 | 4.4704 m/s^2 | True |
| Finite rollout | True | True |

## 16. CEM reference only

The pre-existing CEM feasibility reference remains 1.756-mm tip error, 4.599-m/s directed speed, 19.615-deg direction error, 1.1175-s duration, PASS. It influenced neither initialization nor training.

## 17. Scientific interpretation and classification

Classification: **LOCAL_FEASIBLE_SUPPORT_LOST**. Stop reason: **LOCAL_FEASIBLE_SUPPORT_LOST**. The experiment completed exactly one authorized local K4 campaign and did not unlock additional modes.

## 18. Runtime and exclusions

Episodes `81920`; updates `240`; training runtime `84.312` s; overall runtime `106.961` s; peak CUDA allocation `153.6` MiB. Production model modified: **NO**. Physics conditioning: **NOT ENABLED**. Protected test: **NOT EVALUATED**. Real hardware: **NOT EXECUTED**.

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Policy:
        Feasibility-Constrained One-Shot SAC

    Context:
        SINGLE CANONICAL

    Execution:
        OPEN LOOP

    Reward:
        rl_whip_reward_v2
        WEIGHTS UNCHANGED

    Failed-rollout evaluation:
        HARD_SAFETY_FAILURE_QUARANTINE

    Initial policy center:
        5B4_LEVEL1_SAC

    CEM initialization:
        NO

    Active temporal modes:
        K = 4

    Active stochastic dimensions:
        12

    Initial spectral scale:
        0.05

    Allowed spectral scale:
        [0.005, 0.10]

    Duration:
        FIXED 1.20 s

    Safety critic:
        ENABLED

    Target stochastic failure rate:
        20%

    Episodes:
        81920

    Initial feasible rate:
        49.37%

    Final stochastic feasible rate:
        13.82%

    Feasible & progress>=0.25:
        1.81%

    Feasible & progress>=0.50:
        0.10%

    First stochastic scientific success:
        NONE

    Highest deterministic level:
        2

    Deterministic progress:
        0.512411

    Tip distance:
        658.778 mm

    Directed speed:
        1.017 m/s

    Direction error:
        55.005 deg

    UAV displacement:
        0.4290 m

    Feasible:
        YES

    Scientific task:
        FAIL

    Result:
        LOCAL_FEASIBLE_SUPPORT_LOST

    Production model modified:
        NO

    Physics conditioning:
        NOT ENABLED

    Protected test:
        NOT EVALUATED

    Real hardware:
        NOT EXECUTED
