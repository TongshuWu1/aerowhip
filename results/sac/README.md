# SAC results

This is the retained pure-SAC negative baseline. It uses the same full
production UAV/residual/DDER simulator family and a sequential six-dimensional
control action, with no CEM or demonstrations.

The run was stopped after 1,206,272 of 3,000,000 requested episodes because it
had only 106 successes (0.0088% cumulative success) and no new success in the
last 112,640 episodes. Its final mean maximum UAV displacement was 1.46 m. The
last periodic checkpoint was written at 1,126,400 episodes; the CSV and stopped
summary contain the later observations through the actual stop.

Contents:

- `plots/`: success/throughput and learning-diagnostic figures.
- `data/`: training log, configuration, status, and stopped-run summary.
- `checkpoints/`: the last durable periodic checkpoint for reproducibility.
