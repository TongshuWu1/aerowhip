# Milestone 7C — CEM Teacher Robustness-Basin Audit

## 1. Why this audit was run

Milestone 7B found that exact CEM actions replayed 252/252 successes, but cubic-spline approximations retained 0/252 and a train-only MLP with roughly 0.01 normalized coordinate error retained only 35.81% physical success. This audit tests whether the CEM teachers themselves occupy narrow success basins.

## 2. Scope and prohibitions

This was a simulator/action-support diagnostic using existing artifacts only. New CEM solves: **0**. Learning: **none**. Diffusion, scorer, SAC, final TEST, protected `fig8vertical_002`, theta randomization, and hardware were not used.

## 3. Frozen production contract

The model remained `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`. Every action used the normalized production 49-D decoder, variable active duration, analytic 0.30-s settle, hold, 2.40-s evaluation horizon, and fixed-2048 CUDA physics. All hard scientific gates were unchanged.

## 4. Teacher set

- Designed Milestone-6A contexts: 256
- Exact authoritative successful teachers: 252
- Unique physical state IDs: 187
- One verified action/context: yes
- Exact replay before perturbation: 252/252 PASS

## 5. Perturbation design

Each teacher was evaluated exactly plus 32 deterministic samples at every nonzero level:

- IID normalized acceleration-coordinate noise: 0.001, 0.0025, 0.005, 0.01, 0.02
- Temporally smoothed acceleration noise at the same scales
- Duration-only errors: ±1, ±2, ±5, ±10 ms

All perturbed actions were canonicalized by the production-equivalent coordinate and radial bounds before physics. Total authoritative rollouts: 113,148.

## 6. Acceleration-noise survival

| σ | IID survival | Smooth survival | IID feasible | Smooth feasible |
|---:|---:|---:|---:|---:|
| 0.0010 | 37.88% | 34.14% | 68.51% | 64.05% |
| 0.0025 | 32.03% | 23.28% | 61.37% | 53.25% |
| 0.0050 | 20.87% | 10.31% | 51.45% | 41.89% |
| 0.0100 | 7.61% | 1.70% | 38.36% | 30.99% |
| 0.0200 | 1.28% | 0.21% | 28.81% | 20.87% |


## 7. Duration sensitivity

| Absolute duration error | Success survival | Feasibility |
|---:|---:|---:|
| 1 ms | 40.08% | 70.63% |
| 2 ms | 36.90% | 67.66% |
| 5 ms | 33.53% | 63.29% |
| 10 ms | 24.60% | 55.36% |


## 8. Discrete robustness radii

The reported radius is the largest tested perturbation level retaining the stated per-context survival probability; it is not an interpolated continuous certificate.

- Median IID r90: 0.0000
- Median IID r50: 0.0000
- Median smooth r90: 0.0000
- Median smooth r50: 0.0000
- Median duration r50: 5.0 ms

## 9. Relation to learned policy error

The development-selected residual MLP has median normalized acceleration-coordinate RMSE 0.04394 and scientific success 26.98%. Only 0.00% of its outputs lie within the corresponding teacher's discrete IID r50.

The train-only memorization control reduces median acceleration RMSE to 0.00115, yet success is 35.81%; only 15.35% lie within teacher IID r50. This directly connects small coordinate error to physical failure.

## 10. Hard-gate behavior

At σ=0.01, IID gate pass rates were:

- tip_first: 29.92%
- impact_window: 48.44%
- tip_distance: 29.92%
- directed_speed: 31.06%
- direction: 33.20%
- uav_displacement: 38.48%
- uav_speed: 97.72%
- command_acceleration: 100.00%
- finite: 100.00%


The joint scientific-success rate is lower than most individual gate rates because all timing, strike, direction, and feasibility requirements must hold simultaneously.

## 11. Group heterogeneity

At IID σ=0.01:

- A_TARGET: 12.30%
- B_STATE: 8.64%
- C_JOINT: 5.24%
- D_EDGE: 4.08%


## 12. Root-cause classification

**TEACHER_MANIFOLD_KNIFE_EDGE**

- IID survival at σ=0.005: 20.87%
- Smooth survival at σ=0.005: 10.31%
- IID survival at σ=0.01: 7.61%
- Smooth survival at σ=0.01: 1.70%

Nominal CEM actions occupy narrow physical success basins. The next authorized method should be a small 16-32-context robust-CEM pilot that optimizes perturbation survival and hard-gate margins before any policy training.

## 13. What should not happen next

Do not resume the broad 768-context teacher campaign, do not change policy architecture, and do not train another action-coordinate imitation model on the same nominal labels. A robustness-aware teacher pilot must succeed first if the classification supports it.

## 14. Restrictions

- Production model modified: **NO**
- New CEM solves: **0**
- Learning: **NONE**
- Final TEST: **NOT EVALUATED**
- Protected test: **NOT EVALUATED**
- Hardware: **NOT EXECUTED**

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Existing teachers:
        252

    New CEM solves:
        0

    Learning:
        NONE

    Authoritative perturbation rollouts:
        113148

    Exact teacher replay:
        252 / 252 PASS

    IID survival at sigma=0.005:
        20.87%

    IID survival at sigma=0.01:
        7.61%

    Smooth survival at sigma=0.005:
        10.31%

    Smooth survival at sigma=0.01:
        1.70%

    Median IID discrete r50:
        0.0

    Median smooth discrete r50:
        0.0

    Development-selected MLP median acceleration RMSE:
        0.04394

    Train-memorization median acceleration RMSE:
        0.00115

    Root cause:
        TEACHER_MANIFOLD_KNIFE_EDGE

    More CEM data justified:
        ROBUST-CEM PILOT ONLY; NO BROAD DATASET CAMPAIGN

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
