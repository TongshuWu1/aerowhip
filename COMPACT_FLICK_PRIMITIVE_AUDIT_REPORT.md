# Compact Flick Primitive Audit — Decision Report

## Outcome

The deliberately simple two-pulse approach is physically meaningful but not sufficient under the current fixed-timing formulation.

- Shared-axis 5-D family: **0/6** scientific successes.
- Independent-axes 7-D nested repair: **0/6** scientific successes.
- The 7-D repair consistently achieves safe, tip-first target entry: **6/6**.
- Its best directed entry speed is **1.2787 m/s**, far below the required **4.0 m/s**.
- No learning was performed and no scientific/model contract was changed.

## What was learned

The old high-dimensional learning attempts mixed together three questions: whether the action representation contains a solution, whether exploration can find it, and whether a policy can learn it. This audit isolates the first question.

The 5-D family could occasionally reach the target but could not control its arrival direction. Giving the reverse pulse its own direction fixed that part: two of six 7-D maneuvers passed the 30-degree direction gate, and all six entered tip-first. The remaining failure is consistent across seeds and is specifically arrival speed.

This means we should **not train a policy on this primitive yet**. The next decision is representation-level, not algorithm-level.

## Current production path

    measured/settled state + target + direction
        -> compact physical maneuver parameters
        -> 16 acceleration knots
        -> normalized production action [49]
        -> production decoder
        -> ACTIVE command
        -> analytic 0.30 s SETTLE
        -> stationary HOLD
        -> frozen UAV + residual + DDER simulator
        -> unchanged scientific hard gates

For these audits, bounded CEM optimizes only the compact parameters. It is an offline support oracle; no policy, critic, replay, entropy, diffusion, scorer, or behavior cloning is involved.

## Recommended next decision

If continuing, make exactly one additional nested representation test: add a learned pulse-switch fraction to obtain an 8-D primitive. The evidence suggests the fixed 50/50 timing causes the tip to arrive during HOLD after its useful high-speed phase. Do not add learning until production CEM finds repeatable hard-gate success in that family.

Detailed results:

- [Five-parameter report](FIVE_PARAMETER_FLICK_FEASIBILITY_REPORT.md)
- [Seven-parameter report](SEVEN_PARAMETER_FLICK_FEASIBILITY_REPORT.md)

