# PPO results

The retained checkpoint is the pure PPO terminal checkpoint trained from a
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
