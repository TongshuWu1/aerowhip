# Seven-Parameter Independent-Axes Flick Feasibility Report

## Scientific question

Does the minimal nested repair—independent azimuth/elevation for the reverse pulse—add enough control to make the compact two-pulse family scientifically successful?

The seven parameters are first-pulse azimuth/elevation, reverse-pulse azimuth/elevation, two pulse magnitudes, and maneuver duration. The fixed half-time switch, smooth `sin^2` pulse shapes, production 49-D action codec, frozen physics, 0.30 s analytic settle, 2.40 s evaluation horizon, fixed-2048 numerical contract, and scientific gates were unchanged.

## Exact evaluation pipeline

1. Project seven physical parameters into their fixed bounds.
2. Construct two target-relative unit axes.
3. Decode two smooth `sin^2` acceleration lobes at the production 16 knots.
4. Encode knots and duration into the normalized production action `[49]`.
5. Decode through the same production action codec used by CEM and deployment.
6. Generate ACTIVE -> 0.30 s SETTLE -> HOLD FullState commands.
7. Run full UAV, residual, and 12-node DDER physics with the fixed numerical batch of 2048.
8. Evaluate the unchanged hard scientific gates.
9. Rank scientific success first, then feasible tip-first entries by speed/direction gate deficit, then other feasible candidates by true minimum tip distance.
10. Replay the selected action authoritatively as logical batch one with cyclic padding to 2048.

The support-specific ordering is local to this representation audit. It does not change the production task reward or simulator.

## Initialization

The seven-dimensional family strictly contains the five-dimensional one. Its mean was the exact seed-47 5-D near-hit embedded losslessly:

- first axis: the original shared axis;
- reverse axis: the exact negation of that axis;
- magnitudes and duration: unchanged.

A regression test verifies that both parameterizations produce identical 16-knot commands at initialization. This prevents the audit from becoming a second global rediscovery test.

## Six-seed authoritative result

| Seed | PASS | Feasible | Min tip | Entry tip | Entry speed | Entry angle | UAV disp | UAV speed | Max accel | Duration | Segment |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 42 | NO | YES | 31.875 mm | 47.517 mm | 1.0149 m/s | 28.460 deg | 0.4079 m | 1.4813 m/s | 9.3905 m/s² | 0.6922 s | HOLD |
| 43 | NO | YES | 29.208 mm | 47.610 mm | 1.2455 m/s | 36.104 deg | 0.4582 m | 1.5108 m/s | 10.5143 m/s² | 0.6776 s | HOLD |
| 44 | NO | YES | 33.504 mm | 46.313 mm | 1.2787 m/s | 39.702 deg | 0.4499 m | 1.6186 m/s | 9.6604 m/s² | 0.7284 s | HOLD |
| 45 | NO | YES | 32.675 mm | 46.682 mm | 0.9458 m/s | 26.859 deg | 0.4043 m | 1.4450 m/s | 10.1096 m/s² | 0.7069 s | HOLD |
| 46 | NO | YES | 30.450 mm | 47.163 mm | 0.9315 m/s | 37.479 deg | 0.3802 m | 1.5156 m/s | 10.5468 m/s² | 0.7062 s | HOLD |
| 47 | NO | YES | 46.348 mm | 47.893 mm | 0.8681 m/s | 29.535 deg | 0.4233 m | 1.6359 m/s | 12.4959 m/s² | 0.6944 s | HOLD |

Authoritative scientific successes: **0/6**. Population scientific successes: **0 across 221,184 rollouts**.

All six selected actions were finite, feasible, tip-first, and entered the target sphere. Independent reverse direction therefore fixed the gross placement/direction problem. It did not create the required directed crossing speed: the best entry speed was 1.2787 m/s, leaving a 2.7213 m/s deficit to the 4.0 m/s gate. No seed passed both the speed and direction gates.

## Interpretation

This is a clean representation result:

- command generation is healthy;
- production replay is healthy;
- target placement support is strong;
- safety is not the observed failure;
- fixed-half-time two-pulse velocity support is not demonstrated.

The result does **not** justify SAC, diffusion, a scorer, or more training data. A learner cannot recover a successful action from a family that the production optimizer did not find after a hard-gate-directed six-seed search.

The next smallest scientific question, if authorized later, is whether one additional timing degree of freedom—the pulse switch fraction—can align target entry with the cable's high-speed phase. That would be an eight-parameter nested family, not a new learning system.

## Verification and artifacts

- Contract/regression tests: **18 passed**.
- Seed 42 controlled run: `data/planning/seven_parameter_flick_audit_v1/2026-08-31T045258.377527Z`
- Seeds 43–47 controlled run: `data/planning/seven_parameter_flick_audit_v1/2026-08-31T045600.083023Z`

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Learning:
        NOT USED

    Primitive:
        SEVEN-PARAMETER INDEPENDENT-AXES TWO-PULSE FLICK

    Production action representation:
        NORMALIZED [49]

    CEM population rollouts:
        221,184

    Canonical authoritative CEM:
        0 / 6 seeds PASS

    Feasible selected maneuvers:
        6 / 6

    Tip-first target entries:
        6 / 6

    Best entry directed speed:
        1.2787 m/s

    Required directed speed:
        4.0 m/s

    Family support:
        NOT DEMONSTRATED

    Classification:
        SEVEN_PARAMETER_FLICK_SUPPORT_NOT_FOUND

    Production model modified:
        NO

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED

