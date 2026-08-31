# Milestone 7B — Structured CEM Residual Policy Report

## 1. Decision and scope

The diffusion/scorer branch was retired from the active experiment because increasing the frozen generator from 32 to 1,024 candidates still achieved only 60.94% oracle success on heterogeneous states. This run tests the simpler question directly: can one deterministic network interpolate a consistent family of authoritative CEM maneuvers? It ran **zero new CEM solves**, no diffusion, no scorer, no SAC, no protected test, and no hardware.

## 2. Current production pipeline

The active scientific path is:

1. A physically propagated UAV/cable state is converted to the fixed root-centered, yaw-aligned 83-D `PolicyContext`.
2. The goal rotates one canonical Milestone-6A maneuver family into the local strike direction. This is the deterministic target-conditioned center.
3. A small MLP reads the normalized 83-D context and predicts one residual around that center.
4. The selected maneuver coordinate is decoded to the authoritative normalized 49-D production action: 16 three-axis acceleration knots plus maneuver duration.
5. The unchanged decoder generates `ACTIVE -> 0.30-s analytic SETTLE -> HOLD`, with a 2.40-s evaluation horizon.
6. The unchanged frozen UAV/residual/DDER simulator runs through the fixed 2,048-row numerical contract.
7. The unchanged hard scientific gates decide success.

There is one network query, one returned action, and one open-loop execution. CEM appears only offline in the already-existing teacher artifacts.

## 3. Existing teacher data

- Source contexts: 256
- Authoritative successful contexts used: 252
- Unique physical state IDs: 187
- Labels/context: exactly one
- Action representation: normalized production 49-D
- Multi-solution partial-7A labels used: no

The one-label rule deliberately avoids averaging several CEM modes for one context.

## 4. Frozen scientific contract

Model: `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`. The production action codec, 0.45–1.80-s active duration, smooth settle, 2.40-s observation horizon, fixed-2048 CUDA path, and all hard gates were unchanged.

## 5. Primitive reconstruction audit

Before learning, every teacher and every spline reconstruction was replayed through authoritative physics. The exact teacher replay reproduced 252/252 successes.

| Control points/axis | Latent dim | Replay successes | Retention | Action RMSE | Gate |
|---:|---:|---:|---:|---:|:---:|
| 4 | 13 | 0/252 | 0.00% | 0.24001 | FAIL |
| 6 | 19 | 0/252 | 0.00% | 0.21990 | FAIL |
| 8 | 25 | 0/252 | 0.00% | 0.19149 | FAIL |


Selected regression representation: **FULL_49D** (49 dimensions). The smallest spline was accepted only if it retained at least 90% of the original authoritative successes; otherwise the full 49-D production action was retained.

## 6. State-disjoint split

- Training states: 150
- Development states: 37
- Training contexts: 215
- Development contexts: 37
- State overlap: none
- Canonical state: training only
- Final 7A TEST: not evaluated

## 7. Deterministic residual policy

The policy is an 83 -> 256 SiLU -> 256 SiLU -> 49 MLP with 99,889 parameters. Its output layer starts at zero, so initialization exactly reproduces the rotated CEM-family center. It was trained with Huber regression on the teacher residual, AdamW at 3e-4, and state-disjoint early stopping. Best update: 100; best development Huber: 0.001228.

## 8. One-action authoritative comparison

| Method | Train success | Development success | Development feasible |
|---|---:|---:|---:|
| Target-conditioned CEM warm start | 7.44% | 2.70% | 13.51% |
| Nearest-neighbor retrieval | 30.70% | 8.11% | 43.24% |
| Deterministic residual MLP | 28.84% | 16.22% | 70.27% |
| Train-only memorization control | 35.81% | diagnostic only | diagnostic only |

These are not action-space claims: every number is one decoded action replayed in the authoritative production simulator.

## 9. Physical development metrics for the residual MLP

- Median minimum tip distance: 57.30 mm
- Median directed speed at best event: 4.315 m/s
- Median direction error: 29.62 deg
- Median UAV displacement: 0.4797 m
- Median UAV speed: 2.4046 m/s

## 10. Conditioning checks

Target-conditioned correct-action success was 67.19%; target-swapped action success was 14.06%. State-conditioned correct-action success was 32.81%; state-swapped action success was 10.94%. These controlled swaps keep the original physical context and change only which predicted action is executed.

## 11. Interpretation

Even the bounded train-only memorization control did not reproduce half of its training maneuvers in physics. Additional CEM labels are therefore not justified; the regression representation/objective must be reviewed.

## 12. Active versus historical code

The production path for this experiment is limited to `PolicyContext`, the fixed normalizer, the deterministic residual policy, the authoritative 49-D codec, smooth command continuation, and the production simulator. Historical SAC, diffusion, scorer, and experimental structured-FiLM modules were not imported by the deployment policy and were not deleted; they remain only for reproduction of earlier negative results.

## 13. Restrictions

- New CEM solves: **0**
- Diffusion/scorer/SAC: **NOT USED**
- Production model modified: **NO**
- Final TEST: **NOT EVALUATED**
- Protected `fig8vertical_002`: **NOT EVALUATED**
- Hardware: **NOT EXECUTED**

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Method:
        DETERMINISTIC CEM-FAMILY RESIDUAL POLICY

    New CEM solves:
        0

    Teacher contexts:
        252

    Teacher labels/context:
        1

    Selected representation:
        FULL_49D

    Latent dimension:
        49

    Training contexts:
        215

    Development contexts:
        37

    Warm-start development success:
        2.70%

    Nearest-neighbor development success:
        8.11%

    Residual-MLP training success:
        28.84%

    Residual-MLP development success:
        16.22%

    Residual-MLP development feasibility:
        70.27%

    Train-only memorization success:
        35.81%

    Result:
        DETERMINISTIC_POLICY_NOT_FITTING_TEACHERS

    More CEM data justified:
        NO

    Online CEM:
        NO

    Policy query count:
        1

    Execution:
        OPEN LOOP

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
