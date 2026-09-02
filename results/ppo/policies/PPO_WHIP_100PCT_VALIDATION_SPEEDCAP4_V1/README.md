# PPO Whip — 100% Validation, Speed-Reward Cap 4 m/s

This directory freezes the successful PPO result from:

`data/policy_training/whip_ppo_directional_components_speedcap_d15_i2_time1_7s_2m_v1/2026-09-01T052155.789340Z`

The policy uses the production UAV, causal residual, and 12-node DDER model. It
is a 10 Hz closed-loop sequential PPO controller with a seven-second horizon.
The directed-speed shaping reward is capped at 4 m/s; the scientific task gate
is unchanged.

The selected checkpoint at 614,400 training episodes achieved 64/64 validation
success. Its mean maximum UAV displacement was 1.078 m, so this is preserved as
a strong task-success baseline rather than a compact-motion solution.

## Frozen checkpoints

- `checkpoints/best_validation.pt`: selected 64/64 checkpoint.
- `checkpoints/terminal.pt`: terminal 2,000,896-episode checkpoint.

No later compactness training may overwrite these files.
