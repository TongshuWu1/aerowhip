# PPO whip reward studies

These compact folders preserve the configuration, status, validation summaries,
CSV history, and main figures for recent displacement/return experiments. They
intentionally omit redundant failed checkpoints and launcher logs.

- `forward_return_100`: doubling only the terminal return bonus produced almost
  no behavioral change.
- `terminal_displacement_d60`: produced more return and lower terminal
  displacement, but validation success fell; best guarded checkpoint was
  461/512 = 90.04%.
- `dense_return_release_50`: established a better-shaped dense mechanism reward;
  471/512 = 91.99% at its best checkpoint.
- `blended_progress_fresh_failed`: 50/50 world/attachment progress collapsed
  after an early 45.90% peak and was stopped.
- `attachment_progress_fresh_failed`: pure attachment-compensated progress found
  a runaway-translation exploit; latest validation reached 0/512 with about
  49.5 m mean displacement.
- `attachment_progress_warm_pilot`: controlled warm-start diagnostic showing
  only modest mechanism change.

The selected D50 and retained dense-return policies are stored separately under
`results/ppo/policies/`.
