# Initial-state-only deployment training

The actor plans in a nominal model from one initial estimate. Planning saves a
finite force sequence, truncated at the first predicted valid hit (including
inside a 100 ms action hold). All commands and per-trial cutoffs are fixed before
the separate plant runs. Plant motion or contact cannot extend, repeat, or
shorten the sequence. The original hover PID takes over at its scheduled end.

Training and deterministic validation both use this path. The ordinary task
environment accumulates first-hit rewards, while the physical plant continues
through the remaining commands and recovery. A refused plan receives nominal
progress reward for learning, but is never counted as an executed hit. PPO gets
the whole attempt reward at the final planning action, with gamma and GAE lambda
both 1. Tests check that this return credits every planning action.

## Reward and validation

- Elapsed-time cost: **1 point/s**, charged for planned strike duration.
- PID recovery: at most **10 points** deducted if it never satisfies the live
  settling criterion within 10 seconds; no per-second recovery penalty.
- Hit gates and existing strike reward weights are unchanged: target radius
  5 cm, directed world tip speed 4 m/s, angle at most 45 degrees. Angle is only a
  success gate. Displacement integral weight remains zero; terminal weight 40.
- Every update uses a fixed 256-trial validation set. The guard checks overall
  hit, plan and hit-plus-recovery rates, nominal performance, reward and point
  cost. Rejected updates restore weights and optimizers. Best-checkpoint
  selection prioritizes hit plus recovery, then reward.
- After training, the selected checkpoint is evaluated on 512 further trials
  using a separate seed, saved as `deployment_holdout.json`.

## Provisional uncertainty

25% of trials use the exact nominal hanging state and dynamics. Other trials
sample uniform perturbations independently per coordinate where applicable:

| Quantity | Range |
| --- | --- |
| Known initial root position | ±1 cm |
| Known initial root velocity | ±0.015 m/s |
| Known initial cable tilt | ±1 degree per horizontal axis |
| Known initial cable angular velocity | ±0.03 rad/s |
| Root state estimation error | ±2 mm, ±0.01 m/s |
| Cable tilt estimation error | ±0.2 degree per horizontal axis |
| Cable stiffness and damping | ±10% |
| Applied force gain | ±2% per axis |
| Applied force first-order lag | 0–10 ms |

Cable rotations preserve segment lengths. Force gain and lag are applied to
both strike execution and PID recovery, with the same 3.2 N command envelope.
These are engineering assumptions for simulation robustness, **not calibrated
hardware distributions**. No attitude or motor-allocation model has been added.

## Fresh experiment boundary

Previous exploratory checkpoints, training measurements and replays were
cleared on 2026-09-05. The retained preliminary recordings and calibrated cable
parameters establish the physical baseline; they are not policy-deployment
failure data. There are no retained PPO/SAC performance results for the paper.

Refused plans count against overall success. Different evaluation batch sizes
currently produce different seeded scenarios. For a paper comparison, save an
explicit common validation scenario set and reserve a separate test set.
See [the paper experiment protocol](20260909-doc-cleanup/PAPER_EXPERIMENT_PROTOCOL.md).
