# DDER-MPPI settings profiles

Save versioned online-controller settings profiles in this directory.

The online UI opens this folder by default for both **Load profile** and
**Save profile**. Each JSON profile records the cable-model and warm-start
paths, strike task, controller settings, and hidden plant-truth ratios.
New profiles also record whether validated between-strike EI/Cb adaptation is
enabled. Older profiles remain compatible and preserve their original
fixed-model behavior by defaulting adaptation to disabled.

Profiles store configuration only. Executions and predicted trajectories remain
under `data/drone_mpc/receding_mppi/`.
