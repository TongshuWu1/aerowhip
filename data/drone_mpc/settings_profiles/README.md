# DDER-MPPI settings profiles

Save versioned online-controller settings profiles in this directory.

The online UI opens this folder by default for both **Load profile** and
**Save profile**. Each JSON profile records the cable-model and warm-start
paths, strike task, controller settings, and hidden plant-truth ratios.

Profiles store configuration only. Executions and predicted trajectories remain
under `data/drone_mpc/receding_mppi/`.
