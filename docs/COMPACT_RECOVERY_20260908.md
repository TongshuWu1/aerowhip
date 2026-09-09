# Reduced recovery climb

The user reported continued climb and a slow return after the whip in the
FullState rehearsal. The latest saved example was
`runs/rehearsals/20260908-181353-400309`, using the selected older native policy
`20260908-152029-040639-seed655/checkpoints/best_validation.pt` at target
[1, 0, 1.4] m. This is separate from the current farther-target PPO training.

Its one-second whip ends with reference height 2.4061 m and upward velocity
1.6896 m/s. Braking at only 1.5 m/s² allowed another 1.02 m of commanded climb.
The vertical reference also waited for horizontal braking before beginning
the return. The fitted execution prediction peaked at 3.4449 m.

Native FullState exports now use `deployment/compact_recovery.py`:

- Retain the exact terminal P/V/A and every original whip packet.
- Smoothly transition vertical acceleration over 0.2 s toward braking bounded
  by 3.5 m/s², then smoothly release that braking over 0.2 s.
- Begin vertical return as soon as its braking finishes, independently of the
  horizontal braking/return. Horizontal braking keeps its previous 0.3 s ramps
  and 2.5 m/s² bound.
- Derive all P/V/A from analytic trajectories; no acceleration clipping or
  abrupt velocity reset. Allocate return speed/acceleration budgets between XY
  and Z so their combined bounds remain 0.4 m/s and 0.3 m/s² after braking.
  Vertical return uses at most 0.3 m/s and 0.2 m/s². Lengthen return durations
  to satisfy these bounds, then retain the three-second final hold.

These bounds are simulation reference choices, not measured hardware limits.
Some rise remains because the preserved whip ends with upward momentum.
Recovery prediction is still outside the model's empirically validated scope.

## Verified result

New saved rehearsal: `runs/rehearsals/20260908-182453-378266`.

| Quantity | Previous | Updated |
|---|---:|---:|
| Maximum commanded recovery height | 3.4257 m | 2.9275 m |
| Maximum predicted recovery height | 3.4449 m | 3.0797 m |
| Commanded descent starts, from whip start | 2.3172 s | 1.6516 s |
| Complete CSV duration, including recovery | 16.5000 s | 15.7333 s |

GPU reproduction used the same checkpoint bytes, start and target. Every whip
command and predicted whip cable/origin sample matched exactly; the valid hit
was preserved. Full recovery prediction completed, full CSV passed the existing
reference feasibility envelope, and training/export prefix discrepancy was
zero. Original exports remain untouched. **28 targeted tests passed**, including
analytic derivative checks, continuity at both axes' boundaries, combined
return limits, lower peak, downward/stationary exits, exact whip preservation
and legacy export regressions. Tested Windows / RTX 4080.

Audit and comparison plot: `runs/audits/20260908-compact-recovery/`.
The legacy `gentle_recovery.py` behavior is preserved; only native FullState
`complete_packets` selects the new planner. PPO's recovery-excluded objective,
model, residuals and frozen worker are unchanged. No training restart is needed.
Generate a new rehearsal or load the new saved folder to see the updated path;
existing saved replays keep their original trajectory.
