> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Why the 0.4-second MPPI run missed

The current result is not a like-for-like PPO comparison. There are also
exploration and objective issues worth correcting. This audit changed no
active settings, model assets, saved plans, task rewards or limits. Fitting
and PPO remain stopped. User clarification is pending on whether0.4s means
the whole whip or a rolling lookahead for a longer offline maneuver.

## Task and comparator

Current MPPI performs finite-horizon, open-loop trajectory optimization:
`PVAEnvironment.steps=round(duration_s*30)`. It resets to settled hover for
every candidate batch and scores only12 actions/0.4s. It does not advance
the simulated state and shift/reoptimize a0.4s window over a longer maneuver.
There is no terminal value estimate for events beyond that window.

Historical normalized-M1 PPO rehearsal
`runs/rehearsals/20260909-030657-671710` has a1.0s whip and a predicted valid
hit at0.873333s. Its immutable ghost's closest sampled tip distance in its
first0.4s is1.179153m, versus0.039266m over the full whip. Thus even this
successful PPO forecast has not hit by0.4s. These figures compare saved
trajectories, not controlled optimizer performance.

That PPO uses historical virtual force generation/FullState execution, not
the new bounded-jerk/PVA action contract. The only new PVA PPO run is the
16-attempt smoke test20260909-033109-123778; there is no trained PVA PPO
baseline. Matching initial/target coordinates alone does not establish parity.

## Exploration

Read-only paired batches used the same1024 correlated Gaussian draws at four
scales, same0.4s task, fitted40ms delay, reward, model and limits:

| Latent noise | Feasible candidates | Best feasible tip distance | Hits |
|---|---:|---:|---:|
|0.05 (active)|1024|1.254507m|0|
|0.2|1014|1.138925m|0|
|0.5|585|1.088437m|0|
|1.0|217|1.085650m|0|

These are single batches, not optimized or independently validated policies.
The best trajectory saved by the actual25-iteration run reaches1.231647m.
Its final reference displacement is[-0.014740,-0.016289,+0.108080]m: mostly
an upward movement, not a developed forward whip. Initial tip distance is
1.309067m. Wider exploration finds better candidates, but no hit in this
experiment; that is not a proof of global infeasibility within0.4s.

## MPPI has an additional control preference

The code computes weights proportional to `exp(R(z)/temperature) * p(z)/q(z)`.
The base density p is a fixed zero-mean correlated Gaussian with the same
noise scale as the proposal. Consequently the weighted target distribution
contains a zero-action preference, beyond the explicit jerk term in R.
The equivalent negative-log-prior contribution, up to constants, is
`temperature/2 * ||whiten(z)/noise_std||²`. This is in latent tanh coordinates,
not a literal extra line item subtracted from the saved raw reward.

For the actual saved plan that energy is32.9920 reward units, while its
explicit integrated jerk reward penalty is0.0000684651. These quantities
illustrate the different scales; they are not an exact additive decomposition
of the reported stochastic optimizer return. The Gaussian density ratio is
implemented consistently, but stating that matching the visible PPO reward
alone gives an identical optimization objective was incomplete.

The original information-theoretic MPPI derivation discusses this coupling
and decoupling control cost from temperature in sectionIII-D-2. It also
describes advancing and shifting the control window in Algorithm1:
[Williams et al., 2017](https://arxiv.org/pdf/1707.02342).

A paired40-iteration diagnostic used1024 common draws per condition, noise0.3,
temperature1, with all task rewards and dynamics unchanged. Keeping the
zero-centered density-ratio term achieved1.063381m; removing its zero-centered
pull (current-proposal base, the alpha=1 limiting construction) achieved
1.049244m. Neither condition hit. This isolates an optimization effect;
neither variant was adopted in the active configuration. The diagnostic's
fixed sample budget does not reinstate a production iteration ceiling.

## What this means for the next change

First resolve whole-maneuver duration versus rolling lookahead. If0.4s is a
lookahead, implementing state continuation, pending-command history, proposal
shifting and suitable terminal guidance is a distinct change from shortening
an entire offline whip to0.4s. A short window without guidance may still be
unable to value the setup motion for a later strike. This would remain offline
planning unless actual feedback control is separately requested.

Then make the additional MPPI control preference explicit and tune exploration
separately. Preserve the physical hit criteria and compare with the same model,
action contract and maneuver duration before claiming PPO-level performance.
Increasing only the hit bonus cannot distinguish sampled trajectories that
never hit. The current plateau records stalled search, not successful task
completion or proof of an optimum. Reward changes alone are not established
as the solution by these diagnostics.

Evidence: `runs/audits/mppi-miss-20260909/result.json` and
`prior-comparison.json`. GPU diagnostics ran on Windows/RTX4080. The original
saved plan remains unchanged; no production optimization or training restarted.
