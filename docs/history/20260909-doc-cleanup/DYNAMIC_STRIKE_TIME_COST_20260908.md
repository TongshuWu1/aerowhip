> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Dynamic strike continuation: stronger time cost

On 8 September the user reported slow forward travel followed by a late swing,
and authorized improving the current training. The previous cost, 0.2 reward
points per second, deducted only one point over the entire five-second window.

This amendment changes that coefficient to **10 reward points per second**:
two seconds costs 20 points, three seconds 30, and five seconds 50. Saving one
second is worth ten points when the other reward components are equal. The
first valid hit still earns 250 points. The five-second maximum remains; there
is no hard two-second deadline. This coefficient is a design choice to test,
not an established optimum or a guarantee of whipping motion.

The target, initial-state variation, hit gate, relative-tip-speed shaping,
displacement allowance, M0 physics and both residuals, 30 Hz force/FullState
clocks, action limits, PPO settings and 2,048-environment collection batch stay
the same. See [the original reward design](DYNAMIC_STRIKE_REWARD_20260908.md).

## Time accounting

`T` is the duration of the frozen force/reference sequence selected during
offline planning, including the existing short follow-through and packet
rounding. It excludes prehover and exported recovery. It is **not** the later
hit time predicted by the drone/cable execution model: execution is open loop,
and predicted contact does not interrupt its already planned commands.

The native execution scorer previously extended time charging after a valid
hit but could undercharge after an early invalid contact or reference failure.
The amendment replaces the accumulated time component with exactly `-10*T`
for each deployed attempt and adjusts its total return by the difference.
Hit/failure penalties and other components are preserved. Non-deployed
planning failures retain their nominal planning score and failure penalty.
The legacy 20 Hz scorer and all frozen historical source snapshots remain
unchanged.

The sequence length still comes from the existing virtual planning
first-contact/horizon cutoff. If a family of policies always plans the full
five seconds, its time component is constant; this term only distinguishes
policies once they generate different durations. It encourages shorter useful
plans, but it does not measure approach speed directly or establish a travelling
whip wave. Evaluate valid-hit rate, hit time, planned duration, tip/attachment
velocities and cable motion together. A lower duration obtained by missing or
making invalid contact is not a successful improvement.

## Continuation and reproducibility

Parent `20260908-163449-138254-seed656` was cooperatively stopped at **39,936
saved attempts**, with 50 cumulative exploratory valid hits. Its latest and
terminal checkpoints are preserved. The continuation restores the policy,
value network and both Adam optimizers. It starts a separate reward history,
because old and new reward totals are not comparable. Plateau patience and
minimum additional attempts remain 20,000, measured from the resumed checkpoint;
the 20-evaluation smoothing window and existing tolerances remain unchanged.
The total attempt budget remains 500,000. Checkpoints lack RNG state, so this
is a seeded continuation rather than bitwise uninterrupted stochastic training.

Configuration and previous workspace:
`config/experiments/20260908-dynamic-strike-time10/`.
The amendment records the source checkpoint hash and exact reward overrides.
`prepare_training(..., reward_overrides=..., keep_optimizer_state=True,
keep_stopping_history=False)` saves changes before freezing the new worker.
It rejects changing rewards while preserving stopping history.

Verification artifacts: `runs/audits/20260908-dynamic-strike-time10/`.
`tools/check_time_cost.py` compares the same checkpoint and scenarios with the
parent execution scorer, checks unchanged forces/poses/cable outcomes and
non-time reward components, then runs a bounded full-horizon stochastic PPO
update and checks live-scene reward parity. Tests exercise actual first-contact
scoring, repeated finalization, zero and nonzero coefficients, continuation
immutability and a fresh convergence origin. These are implementation checks
on Windows/CUDA, not evidence of an improved policy or a real-flight result.
