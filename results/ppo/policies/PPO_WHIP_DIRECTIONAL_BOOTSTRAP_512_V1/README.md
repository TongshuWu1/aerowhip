# PPO whip directional bootstrap — frozen

This directory preserves the best deterministic policy from the fresh PPO run
at 174,080 episodes. It is the source policy for compactness continuation and
must not be overwritten.

The authoritative production audit passed 488 of 512 physically propagated
initial states (95.31%). The canonical replay passed at 2.16 s with 10.98 m/s
directed tip speed and 9.23 degrees direction error. Mean maximum UAV
displacement across the 512-state bank was 1.348 m, so this is a strong strike
policy rather than a compact-motion solution.

Contents:

- `checkpoints/best_validation.pt`: frozen policy checkpoint.
- `audit/`: canonical replay and complete 512-state deterministic audit.
- `plots/`: source training and validation curves.
- `freeze_manifest.json`: provenance, metrics, and SHA-256 hashes.

The protected test was not evaluated and hardware was not executed.
