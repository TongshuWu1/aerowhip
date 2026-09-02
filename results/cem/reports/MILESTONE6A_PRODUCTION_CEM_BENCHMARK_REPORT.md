# Milestone 6A — Production CEM Stabilization and Overnight Generalization Benchmark

## 1. Why learning was intentionally paused

This milestone isolates the planner. SAC, critics, replay, entropy, supervised
actors, behavior cloning, and policy amortization were deliberately excluded.
The question is whether variable-duration CEM itself is a reliable one-shot
open-loop maneuver planner once command continuation and deployment replay are
made production-consistent.

## 2. Frozen model confirmation

All physics used `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`: the frozen UAV
gains and residual (including normalization and FIFO), EI/Cb, measured cable
and attachment geometry, 12-node DDER topology, CUDA float32, PCG32, three
DDER substeps, and four projections. No parameter was fitted or changed.

## 3. Old post-maneuver discontinuity

The former continuation jumped directly from terminal `v_T`/`a_T` to zero
while setting `p_cmd=p_T`. That was a command discontinuity whenever the active
maneuver ended with nonzero velocity or acceleration and could create an
artificial UAV/cable excitation.

## 4. New smooth terminal-settle derivation

For `s=tau/0.30`, the fixed continuation is

    v(s) = (2s^3-3s^2+1)v_T + (s^3-2s^2+s)(0.30)a_T

with acceleration differentiated analytically and position integrated
analytically. The stationary position is not chosen manually:

    p_hold = p_T + 0.5(0.30)v_T + (0.30^2/12)a_T.

This provides continuous boundary values in position, velocity, and
acceleration. Settle acceleration is not clipped and participates in the
unchanged 20 m/s² command-acceleration gate.

## 5. Boundary-continuity verification

- cases: 4;
- maximum measured boundary error: `2.220e-16`;
- finite commands, exact 10-ms time grid, no missing/duplicated sample, and
  continuous yaw: **PASS**.

## 6. New maneuver/evaluation time semantics

- `T_maneuver` is optimized in `[0.45, 1.80] s`;
- `T_settle=0.30 s` is deterministic;
- `T_evaluation=2.40 s` observes active, settle, and hold motion.

These are numerical planning envelopes, not new scientific success gates.
Every event is labeled `ACTIVE`, `SETTLE`, or `HOLD`.

## 7. Action-representation audit

The sole planner/deployment representation is normalized `[49]`: 16 three-axis
acceleration knots followed by normalized duration. Every candidate is decoded
before physics; the same decoder serves population rollout, storage, future
teacher use, and authoritative replay.

- effective encode/decode round-trip maximum: `0.000e+00`;
- physical-knot re-decode maximum: `7.105e-15 m/s²`;
- duration re-decode maximum: `2.220e-16 s`;
- historical seed-46 round-trip maximum: `0.000e+00`;
- codec audit: **PASS**.

## 8. Root cause of prior 65 -> 32 replay attrition

The earlier teacher population ran complete DDER populations at batch 2048,
but its “batch-one” replay preserved only the UAV residual's internal fixed
shape; the coupled DDER state itself was propagated at logical batch one.
Action conversion also occurred only after CEM. Population and deployment did
not share one complete numerical/action contract. Milestone 6A removes both
differences: all coupled physics is padded to 2048 and all candidates use the
normalized production decoder before rollout.

## 9. Fixed authoritative action/deployment contract

Logical batches from 1 through 2048 are cyclically padded to 2048 before UAV
and cable physics and sliced only after metrics are complete. No CEM-only
command semantics or post-optimization representation conversion remains.

The current production pipeline is:

1. Build one root-centered, yaw-local planning context from the physically
   propagated initial UAV/cable state, target, and desired horizontal strike
   direction.
2. Start every independent solve from the same canonical successful maneuver
   family, rotated only through the standard local target-direction transform.
3. Sample normalized `[49]` candidates: 48 acceleration coordinates plus one
   duration coordinate. Project/encode once into the canonical effective
   normalized representation.
4. Decode every candidate with the production codec to 16 physical
   acceleration knots and `T_maneuver` in `[0.45,1.80] s` before any physics.
5. Generate one complete FullState command: exact integration of the
   piecewise-linear active acceleration, the fixed analytic 0.30-s settle, and
   stationary hold through 2.40 s. Yaw is fixed to query-boundary yaw and
   commanded angular velocity is zero.
6. Cyclically pad the complete coupled rollout to numerical batch 2048. Run
   the one frozen UAV/residual/DDER simulator, then slice back to the logical
   population only after physical metrics are computed.
7. Accumulate the unchanged legacy strike objective and scientific hard gates
   over ACTIVE, SETTLE, and HOLD; the settle acceleration participates in the
   20 m/s² command gate.
8. Update one full-covariance CEM distribution from the 5% elites. Contexts are
   independent and never warm-start from one another.
9. At termination, send the exact final top 32 normalized actions through the
   same authoritative fixed-2048 evaluator. Select a final action only from
   this replay, preferring authoritative scientific successes by the accepted
   optimizer objective, otherwise the best authoritative feasible near-miss.
10. Persist that exact normalized action and both population and authoritative
    metrics. This stored action is the deployment/teacher representation; no
    post-CEM conversion path exists.

## 10. Population vs batch-one consistency diagnostic

The 256-action diagnostic covered safe-low, aggressive, near-CEM, and random actions. Hard-gate mismatch count: **0**. Result: **PASS**.

## 11. Canonical six-seed CEM result

| Seed | Result | Tip mm | Directed m/s | Direction deg | Tip first | UAV disp m | UAV speed m/s | Cmd accel m/s² | T maneuver s | Hit s | Segment | Runtime s |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| 42 | PASS | 28.954 | 4.465 | 18.064 | True | 0.486 | 2.432 | 15.139 | 1.118 | 1.110 | ACTIVE | 34.0 |
| 43 | PASS | 23.816 | 4.393 | 21.443 | True | 0.474 | 2.463 | 14.977 | 1.113 | 1.110 | ACTIVE | 34.7 |
| 44 | PASS | 28.954 | 4.465 | 18.064 | True | 0.486 | 2.432 | 15.139 | 1.118 | 1.110 | ACTIVE | 33.0 |
| 45 | PASS | 28.954 | 4.465 | 18.064 | True | 0.486 | 2.432 | 15.139 | 1.118 | 1.110 | ACTIVE | 33.1 |
| 46 | PASS | 28.954 | 4.465 | 18.064 | True | 0.486 | 2.432 | 15.139 | 1.118 | 1.110 | ACTIVE | 34.3 |
| 47 | PASS | 28.954 | 4.465 | 18.064 | True | 0.486 | 2.432 | 15.139 | 1.118 | 1.110 | ACTIVE | 33.8 |

Canonical authoritative gate: **6/6 PASS**. Required: 5/6.

## 12. Canonical success margins

Positive values are margin inside the unchanged scientific gate.

| Seed | Tip margin mm | Speed margin m/s | Direction margin deg | UAV-displacement margin m | UAV-speed margin m/s | Command-accel margin m/s² |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 21.046 | 0.465 | 11.936 | 0.014 | 0.568 | 4.861 |
| 43 | 26.184 | 0.393 | 8.557 | 0.026 | 0.537 | 5.023 |
| 44 | 21.046 | 0.465 | 11.936 | 0.014 | 0.568 | 4.861 |
| 45 | 21.046 | 0.465 | 11.936 | 0.014 | 0.568 | 4.861 |
| 46 | 21.046 | 0.465 | 11.936 | 0.014 | 0.568 | 4.861 |
| 47 | 21.046 | 0.465 | 11.936 | 0.014 | 0.568 | 4.861 |

## 13. Benchmark context construction

The benchmark contains four deterministic 64-context groups: target variation
from canonical state, state variation at canonical target, joint variation,
and upper-state-distance/target-boundary edge diagnostics. Initial states come
only from the existing physically propagated nominal state bank; cable markers
were never independently perturbed.

## 14. CEM benchmark settings

Each context uses population 4096, 5% elites, full covariance, 4–20 iterations,
fixed-2048 physics, top-32 authoritative final selection, and at most three
deterministic seeds. Every context starts from the same canonical successful
maneuver family rotated only by local target direction. Contexts never
warm-start one another.

## 15. Group A target-only results

All 64 target-only contexts passed on their first seed. The canonical settled
state was fixed while targets were stratified over the Phase-1 local domain.

## 16. Group B state-only results

All 64 state-only contexts passed on their first seed. Initial states span the
existing physically propagated nominal state-bank variation at canonical
target.

## 17. Group C joint results

Joint state/target variation solved 62/64 contexts. Both unsolved cases used
all three deterministic seeds and the complete 20-iteration allowance.

## 18. Group D edge results

Edge variation solved 61/64 on the first seed and 62/64 within three seeds.
The table includes successful-rollout medians and group median planning time.

| Group | N | First-seed | <=3-seed | Median tip mm | Median directed m/s | Median direction deg | Median duration s | Median runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_TARGET | 64 | 100.00% | 100.00% | 28.129 | 4.394 | 24.477 | 1.109 | 33.85 |
| B_STATE | 64 | 100.00% | 100.00% | 32.517 | 4.372 | 25.212 | 1.111 | 33.80 |
| C_JOINT | 64 | 96.88% | 96.88% | 32.349 | 4.310 | 26.504 | 1.114 | 41.79 |
| D_EDGE | 64 | 95.31% | 96.88% | 36.115 | 4.297 | 27.482 | 1.119 | 57.74 |

## 19. First-seed success rate

`98.05%` (251/256).

First-seed misses: D_EDGE_044, C_JOINT_059, D_EDGE_062, C_JOINT_063, D_EDGE_063.

## 20. <=3-seed success rate

`98.44%` (252/256).

Unsolved contexts after all authorized seeds: **4**.

| Context | Group | Seeds attempted | Total iterations | Runtime s |
|---|---|---:|---:|---:|
| C_JOINT_059 | C_JOINT | 3 | 60 | 473.98 |
| D_EDGE_062 | D_EDGE | 3 | 60 | 473.28 |
| C_JOINT_063 | C_JOINT | 3 | 60 | 473.95 |
| D_EDGE_063 | D_EDGE | 3 | 60 | 473.56 |

## 21. Population -> authoritative replay consistency

Population PASS -> authoritative FAIL: `0`.
The assessment uses authoritative deployment replay only; restarts are separated from first-seed results.

Absolute population-to-authoritative differences:

| Metric | Mean | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
| directed_speed_m_s | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| direction_error_deg | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| tip_distance_m | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| uav_displacement_m | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| uav_speed_m_s | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

## 22. Hard-gate robustness margins

Positive values are inside the gate. The fifth percentile exposes knife-edge
passes even when the median is comfortable.

| Gate | Median margin | 5th-percentile margin |
|---|---:|---:|
| Tip position | 17.928 mm | 3.770 mm |
| Directed speed | 0.353 m/s | 0.112 m/s |
| Direction | 4.034 deg | 0.495 deg |
| UAV displacement | 0.012 m | 0.001 m |
| UAV speed | 0.769 m/s | 0.411 m/s |
| Command acceleration | 4.827 m/s² | 3.459 m/s² |

## 23. Maneuver-duration distribution

Successful solutions: count 252, mean 1.113 s, median 1.111 s, p90 1.155 s, p95 1.178 s, maximum 1.277 s.

## 24. Hit-time distribution

Hit time: mean 1.100 s, median 1.100 s, p90 1.149 s, p95 1.170 s, maximum 1.220 s. For `t_hit-T_maneuver`: mean -0.013 s, median -0.015 s, p90 0.019 s, p95 0.077 s, maximum 0.140 s.

## 25. ACTIVE / SETTLE / HOLD strike fractions

- ACTIVE: `74.21%`;
- SETTLE: `25.79%`;
- HOLD: `0.00%`.

## 26. Planning runtime distribution

Runtime: mean 53.33 s, median 34.61 s, p90 76.37 s, p95 105.46 s, maximum 473.98 s. Rollouts/context: mean 26881, median 16544, p95 54728, maximum 247776. Iterations/context: mean 6.50, median 4, p95 13.25, maximum 60.

## 27. Whether duration upper bound became active

`NOT ACTIVE` under the
specified within-0.02-s diagnostic.

## 28. CEM-primary-planner assessment

**CEM_PRIMARY_PLANNER_SUPPORTED**. This is a simulation
planner assessment only, not a real-flight-readiness claim.

## 29. Optional long-horizon frozen-model validation

**NOT COMPLETED.** The mandatory gated CEM experiment was prioritized. Existing
non-protected validation artifacts remain untouched; no fitting was run.

## 30. Recommended next scientific step

Review authoritative success, robustness margins, duration, hit segment, and
runtime. If CEM is supported, decide deliberately whether trial-to-trial
latency is already acceptable or later optimizer amortization is justified.
No learner is started here.

## 31. Explicit exclusions

- SAC: **NOT USED**.
- Actor: **NOT TRAINED**.
- Production model: **NOT REFIT / NOT MODIFIED**.
- Protected test `fig8vertical_002`: **NOT EVALUATED**.
- Real hardware: **NOT EXECUTED**.

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Learning:
        NOT USED

    Planner:
        VARIABLE-DURATION CEM

    Active maneuver duration:
        [0.45, 1.80] s

    Terminal settle:
        0.30 s
        C2 command continuity in p/v/a boundaries as implemented

    Evaluation horizon:
        2.40 s

    Population/deployment action representation:
        IDENTICAL

    Replay consistency:
        PASS

    Canonical authoritative CEM:
        6 / 6 seeds PASS

    Benchmark contexts:
        256

    First-seed authoritative success:
        98.05 %

    <=3-seed authoritative success:
        98.44 %

    Population PASS -> replay FAIL:
        0

    Median tip error:
        32.072 mm

    Median directed speed:
        4.353 m/s

    Median direction error:
        25.966 deg

    Median maneuver duration:
        1.111 s

    Hit segment:
        ACTIVE 74.21 %
        SETTLE 25.79 %
        HOLD 0.00 %

    Duration bound active:
        NO

    Median planning time:
        34.61 s

    P95 planning time:
        105.46 s

    CEM primary planner:
        SUPPORTED

    Long-horizon model validation:
        NOT COMPLETED

    Production model modified:
        NO

    SAC:
        NOT USED

    Amortized actor:
        NOT TRAINED

    Protected test:
        NOT EVALUATED

    Real hardware:
        NOT EXECUTED

Run gate status: `COMPLETED`.

Artifact directory: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\planning\production_cem_benchmark_v1\2026-08-30T072758.266436Z`.
