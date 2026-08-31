# Five-Parameter Shared-Axis Flick Feasibility Report

## Question

Can a smooth, target-relative two-pulse maneuver with only five physical parameters contain the canonical scientific whip? The parameters are shared azimuth/elevation, first-pulse magnitude, reverse-pulse magnitude, and maneuver duration. Both `sin^2` pulses lie on the same spatial line with a fixed half-time reversal.

No learning was used. Every primitive was converted to the production normalized 49-D action and evaluated by the frozen production simulator with the Milestone-6A ACTIVE -> SETTLE -> HOLD command contract.

## Six-seed result

| Seed | Scientific PASS | Feasible | Minimum tip distance | Tip entered first | Entry directed speed | Entry direction error |
|---:|:---:|:---:|---:|:---:|---:|---:|
| 42 | NO | YES | 172.32 mm | NO | — | — |
| 43 | NO | YES | 144.59 mm | NO | — | — |
| 44 | NO | YES | 138.48 mm | NO | — | — |
| 45 | NO | YES | 67.44 mm | NO | — | — |
| 46 | NO | YES | 133.90 mm | NO | — | — |
| 47 | NO | YES | 26.21 mm | YES | 0.913 m/s | 75.01 deg |

Authoritative scientific successes: **0/6**.

Seed 47 proves that this family can safely place `c10` inside the 50 mm target sphere, but the shared-axis reversal cannot simultaneously align and accelerate the arrival: the 4.0 m/s and 30 deg gates both fail.

## Artifacts

- Seed 42: `data/planning/five_parameter_flick_audit_v1/2026-08-31T043118.911654Z`
- Seeds 43–47: `data/planning/five_parameter_flick_audit_v1/2026-08-31T043316.652649Z`

## Classification

`FIVE_PARAMETER_FLICK_SUPPORT_NOT_FOUND`

The result justified exactly one nested repair: preserve the seed-47 near-hit and give the second pulse an independent direction.

