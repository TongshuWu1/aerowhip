# PPO results

## Current controller selection (2026-09-03)

The desktop **Run & Replay** page is pinned to
`policies/PPO_WHIP_FORWARD_REVERSE_RELEASE_D50_V1/checkpoints/terminal.pt`.
This is the current high-success controller: 480/512 = **93.75%** deterministic
success on distinct held-out physically propagated initial states. It is a
10 Hz closed-loop sequential PPO controller for the canonical target and
nominal physics.

`policies/PPO_WHIP_DENSE_RETURN_RELEASE_100_V1` is retained as a deliberate
trade-off candidate. Its final checkpoint returns the UAV about 2.26 cm farther
before impact while preserving 474/512 = **92.58%** validation success. It is
not the default controller.

`reward_studies/` contains compact data and plots for the return/displacement
experiments, including the failed blended and attachment-only progress runs.
Raw timestamped training directories remain locally under `data/policy_training`
and are ignored by Git; the durable evidence needed by another checkout is
curated here.

See [`PPO_WHIP_HANDOFF.md`](../../PPO_WHIP_HANDOFF.md) for the current research
contract, exact results, implementation map, and recommended next step.

## Earlier retained PPO baseline

The earlier retained checkpoint is the pure PPO terminal checkpoint trained from a
random initialization—no CEM action, demonstration, behavior cloning, or prior
policy checkpoint was used.

- Training: 1,001,472 episodes, 100,051 task successes, 9.99% cumulative
  stochastic collection success.
- Nominal deterministic state-bank audit: 469/512 = **91.60%**.
- Exact compiled-command replay from the same initial state: **PASS**, also
  469/512.
- A trajectory compiled for the wrong initial state: 79/512 = **15.43%**.
- Feedback advantage under EI/Cb mismatch: +18.61 percentage points on average.
- Feedback advantage under post-planning disturbances: +17.45 percentage points
  on average.
- Batch-one compilation latency: 12.17 s median, 12.36 s p95.

These results select **10 Hz closed-loop PPO**, not the slower PPO+sim open-loop
compiler, as the current learned control architecture.

The main desktop Simulator tab can now execute this frozen checkpoint once in
the complete production simulator and immediately replay the UAV/cable motion.
This path is simulation-only and does not connect to hardware.

Contents:

- `plots/`: training and validation success curves.
- `data/`: compact run logs, configuration, and summaries.
- `checkpoints/`: terminal and best-validation checkpoints.
- `compiler_audit/`: exact-replay, wrong-state, model-mismatch, disturbance, and
  latency evidence.
- `reports/`: full PPO and compiler technical reports.
