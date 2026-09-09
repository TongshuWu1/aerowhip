# Moving curved recovery after the unchanged whip

The user clarified that recovery should carry momentum into a rounded turn and
descent, rather than stop at the top and slowly return. This supersedes the
independent vertical braking default described in `COMPACT_RECOVERY_20260908.md`.

New native 30 Hz rehearsals use `deployment/curved_recovery.py`. Every whip
command packet is retained exactly, including its endpoint position, velocity
and acceleration. The new tail has two analytic quintics:

1. A moving turn/descent from the whip's terminal P/V/A to an approach point.
   That point is 0.8 m from hover along the horizontal exit direction (position
   displacement is used for an almost vertical exit), at most 0.4 m above hover.
   With no horizontal direction, +X resolves the otherwise ambiguous plane.
2. A slow approach beginning with velocity 0.35 m/s toward hover and zero
   acceleration. It ends at the original hover position with zero velocity and
   acceleration, followed by a three-second hold.

The join is moving, not a stationary braking waypoint. All joins preserve P/V/A;
jerk is finite within each segment but is not constrained continuous across
joins. A stationary exit goes directly to hover without manufacturing a loop.
This is a rounded trajectory, not a fixed-radius aerobatic circle or an attitude
flip. Yaw remains the existing native export convention.

Turn duration is chosen from 1–8 seconds on the 30 Hz clock. Feasible candidates
are ranked by their peak commanded height, then duration. The planner checks
polynomial extrema over the complete continuous segments, not only CSV samples.
Shaping limits include downward acceleration 3.5 m/s², horizontal acceleration
5 m/s², descent speed 1.25 m/s, tilt 45 degrees, total speed 5 m/s and positive
vertical specific force. Existing terminal acceleration, tilt or downward speed
can exceed the shaping defaults; the effective boundary-preserving limits are
recorded. The complete CSV still must pass the policy's saved model envelope.
An infeasible plan is refused. These are provisional simulation limits, not
measured vehicle capabilities or a workspace-clearance guarantee.

## Reproduction of the user's rehearsal

Source: `runs/rehearsals/20260908-181353-400309`, with selected checkpoint
`20260908-152029-040639-seed655/checkpoints/best_validation.pt`. The checkpoint
hash, initial tracked origin and target are identical in all comparisons.
This selected older policy is distinct from the current v3 training run.

| Recovery | Predicted peak height | Commanded peak height | Full CSV duration |
| --- | ---: | ---: | ---: |
| Original stop and return | 3.4449 m | 3.4257 m | 16.5000 s |
| Previous stronger braking | 3.0797 m | 2.9275 m | 15.7333 s |
| Moving curved recovery | 2.7237 m | 2.9634 m | 11.2333 s |

The curved turn lasts 2.0667 s after the one-second whip; the slow approach lasts
5.1333 s. Commanded turn acceleration reaches -3.4344 m/s² vertically and the
largest commanded descent speed is 1.1027 m/s. The reference rises slightly more
than the previous stronger-braking reference, but the fitted model predicts a
substantially lower peak and an earlier descent. This distinction matters: the
numbers describe this model/example, not a guaranteed real-flight improvement.

New complete saved result: `runs/rehearsals/20260908-143530-158718` (directory stamp
uses this machine's local clock). Generate a new rehearsal or open this saved
folder in the native UI. Previous saved replays retain their original paths.

## Verification

- 42 targeted tests passed: polynomial interior extrema, moving join, descent,
  P/V/A continuity and derivative consistency, final hover, stationary/vertical/
  downward exits, refusal of invalid plans, exact whip/clock preservation,
  export packaging, historical recovery and execution-termination regressions.
- Complete fitted drone plus cable GPU rehearsal succeeded on Windows / RTX
  4080. The saved reference passed the existing envelope; the recovery prediction
  completed. Whip commands, predicted tracked positions/rotations, cable frames
  and hit outcome are exactly unchanged. The production export reproduces every
  command, timestamp and phase of the GPU audit exactly.
- Original and previous-recovery artifacts were hash-verified unchanged. Current
  PPO remained running with its frozen sources unchanged. No policy, reward,
  model, residual or optimizer change and no training restart.

Audit: `runs/audits/20260908-curved-recovery/verification.json` and `comparison.png`.
Legacy 20 Hz recovery and the prior planners remain available for historical
interpretation. The UI adds a short description for newly generated native
rehearsals; it does not relabel saved old trajectories as curved recovery.
