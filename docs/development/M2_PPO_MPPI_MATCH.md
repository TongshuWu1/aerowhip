# PPO for the selected single-target M2 whip

A separately authorized fresh run now tests sustained actions and a longer training
budget; see [persistent exploration](M2_PPO_PERSISTENT_EXPLORATION.md). The result
below remains the preserved original one-step PPO baseline.

## Completed result

Training stopped by plateau at 188,416 attempts (92 updates, 647.53 s), not the
500,000-attempt review limit. Best checkpoint: 139,264 attempts. Its independent
replay matches score 1247.6117672260727 and modeled tip entry at 1.087738 s.
The complete 10.4 s CSV/recovery passed and is saved in
`runs/rehearsals_pva/20260911-135019-408009-M2-ppo-whip`. No real PPO flight or
further optimization was run. `completion_check.json` contains independent
NumPy contact timing/speed, source and preservation checks; the final policy
snapshot/preview/recovery are hashed in `final-policy-frozen-manifest.json`.

The desired MPPI-style whip was not learned. The PPO drone is moving forward
at +1.403 m/s at contact and is at [1.195,0.093,1.980] m, nearly above the target.
MPPI is moving backward at -1.517 m/s. PPO gets 1246.52 impact points, 2.72 fold,
17.64 contact-quality and 1.13 cast; MPPI gets 960.96, 432.82, 440.91 and 107.76.
Thus the shared objective already scores MPPI higher (1892.43 versus 1247.61),
despite PPO's larger modeled tip speed (7.51 versus 4.91 m/s).

The current diagnosis is an exploration/local-solution problem: MPPI refines
known whip seeds, while PPO uses a fresh policy with stepwise Gaussian actions
and clipped updates. Release was sampled in only 9 of 92 batches, with no batch
above 4/2,048 release-qualified episodes. The terminal fold preference must
credit a coordinated history, and the policy found a high-speed contact solution
without discovering the better sequence. Plateau stopping after 49,152 attempts
without significant policy-test improvement is not proof of sufficient global
exploration. We have not isolated these causes in a controlled ablation. The
[PPO paper](https://arxiv.org/abs/1707.06347) describes its sampled policy-gradient
surrogate optimization; it does not promise discovering a specified maneuver
from a shared scalar reward. No reward/training change follows this diagnosis.

The following sections record the frozen setup and earlier launch checks.


11 September 2026. The user paused the multiple-target work and explicitly
requested PPO using the same reward and setup as the selected M2 MPPI whip.

Training job: `runs/ppo_pva/20260911-135019-408009`.
Read its `status.json`, `history.json` and `checkpoints/best_evaluation.json`
before reporting progress. Do not start a duplicate. The launch audit is
`runs/audits/M2-ppo-matched-20260911`; its supervisor records the eventual
independent final policy review in `status.json`.

## Matched experiment

- Reference MPPI: `20260910-211435-608306`, the user's selected single-target whip.
- Exact frozen M2-frozen-refit-v1, signature
  `be4bd82c038dad1e2f1508babc9a1c3da121e1f428f2a375942b382bf6a7b0b9`.
  The drone model, distributed cable model and both neural residuals are fixed.
- Initial tracked origin `[0,0,1.255]` m; target `[1.25,0,1.00]` m;
  5 cm virtual sphere; level hover with a hanging, initially motionless cable.
  No start/target randomization or multiple-target objective.
- Same 1.5 s maximum episode, 30 Hz bounded-jerk PVA commands, 150 Hz physics,
  feasibility bounds and integration. PPO has an episode, not MPPI lookahead.
- Exact `reward` and `trajectory_objective` dictionaries and checksummed preferred
  shape reference copied from the selected MPPI job. This includes impact weight
  1600, scale 4 m/s, and no earlier-hit reward or hard minimum hit-speed gate.
- Feasible tip entry defines success. Wave/reversal remain reward preferences
  and diagnostics. Checkpoint selection now explicitly uses
  `tip_contact_then_score_v1`: with this fixed scenario, a feasible deterministic
  hit ranks above a miss; score ranks policies within the same outcome.
  Historical M0 PPO runs retain reward-only selection when the version is absent.

The prior M0 PPO did not include the M2 impact-speed bonus and ranked checkpoints
only by score. Reusing its settings unchanged would not meet this request.
No MPPI command sequence, policy weights or demonstrations initialize training.
The MPPI actions used by the parity audit are test inputs only.

## PPO training and comparison limits

Fresh policy/value/optimizer; seed 655, observation `pva_whip_phase_v3`.
There are 2,048 parallel CUDA environments, a 256-wide network, learning rate
3e-4, four PPO epochs, minibatches of 8,192 and entropy coefficient .005.
Physical simulation is FP64; policy learning is FP32. Fused CUDA ticks and the
existing specialized geometry/linear solvers remain enabled.

Gamma and GAE lambda are 1; training return scale is .01. The existing causal
potential differences telescope to the shared terminal score plus a constant
for this initial state. Feasible scores match MPPI exactly. Infeasible episodes
receive finite -10,000 for PPO learning, whereas MPPI rejects them with negative
infinity. This finite surrogate is an unavoidable practical difference here;
the methods do not have identical optimization or exploration behavior.

Deterministic evaluation runs every four updates. Minimum exploration is 98,304
attempts; plateau patience is 49,152 attempts, using contact-first evaluation and
.005 relative score improvement. The review limit is 500,000 attempts; reaching
that limit is not proof of convergence. These are the existing PPO training
settings, not a new adaptation method. One fixed-scenario 0/100% policy test is
not a real-flight success rate or a robustness estimate. Sampled training hits
are reported separately from the deterministic policy.

## Checks and use

The exact selected MPPI commands scored **1892.4306902448288** in both independent
MPPI and PPO paths. Every component, command packet, success/failure, termination
and minimum distance agreed within the checked tolerance. A fresh 2,048-row
rollout and optimizer update passed finite-value and telescoping-return checks
in 6.94 s. This was discarded as a smoke test, not used as a training warm start.
31 focused tests passed; the checkpoint/preview subset passed again after
correcting preview labels for the tip-contact criterion. Sampled GPU utilization
was 96%; this is an observation, not a theoretical maximum-throughput claim.

The app opens **PPO > Training progress** on this job, with progress bars, policy
tests, infeasibility, curves and saved logs. **Rehearse policy** freezes latest
or best checkpoint without stopping training. Leave **Include recovery / export**
off to inspect an unfinished policy; **Refresh policy rehearsal** loads newly
saved weights. Native Qt/VTK checks verified a live 24,576-attempt snapshot,
correct tip-contact labels and continued training. That early policy was a miss.

The supervisor is already running with training. After normal completion it
freezes and independently replays the best checkpoint, compares its score and
outcome with the recorded evaluation, and attempts complete CSV/recovery export
only if the policy predicts a valid hit. A recovery failure is recorded and does
not become a flight-ready CSV. No further training or retuning is automatic.

The selected MPPI flight, earlier policies, models, raw recordings and forecasts
remain preserved. No fitting, model promotion or physical flight is performed.
The two-target attempts remain evidence, with no further search active.

## Intermediate learned hit, checked while training was active

The deterministic test first reached the target at 65,536 attempts. During
continued training, the best checkpoint frozen at 81,920 attempts independently
reproduced score 1057.7320218029367 and tip contact at 1.1496378185878187 s.
Full recovery passed, producing a 10.4 s CSV and exact-model forecast in
`runs/rehearsals_pva/20260911-135019-408009-M2-ppo-81920-whip`.
See `first-hit-review.json` in the audit for snapshot identity and all metrics.

This is a modeled hit, not an equally good MPPI-style whip: at contact the drone
is still moving forward at 1.42 m/s, with no completed backward release.
The fold component is only 4.83 versus the selected MPPI's 432.82; total score
is 1057.73 versus 1892.43. The higher modeled forward tip speed (5.55 versus
4.91 m/s) does not establish better whipping or real impact force. Training
continues under the unchanged matched reward. This intermediate snapshot does
not replace the final selected policy or the MPPI real-flight selection.
