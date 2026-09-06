# Learning

PPO and SAC share the point-force task and initial-state-only strike execution.
`simple_ppo.py` implements PPO; `simple_sac.py` implements SAC.
`point_force_env.py` provides the 79-value nominal-state observation, force
mapping, hit conditions and reward. `deployment_rollout.py` plans a finite
sequence from an initial estimate, executes it on a separate plant, and scores
its strike and PID recovery. SAC replay integration is in `sac_deployment.py`.

The current action has three world-frame force components. Exploration uses Fx
and Fz; Fy is fixed at zero for the symmetric task. The actual plant does not
feed observations or hit results into force selection or the scheduled cutoff.

The hit conditions require a tip-first target entry within 5 cm, at least
4 m/s of world-frame tip velocity along the desired strike direction, and at
most 45 degrees of velocity-direction error. Angle is a binary gate. The
terminal cable tangent is a diagnostic, not the hit vector.

Task reward and force settings are in `config/ppo.json` and are shared by both
algorithms. The elapsed-time cost is 1 point/s. A failed PID recovery adds a
bounded cost. Whole-plan returns are undiscounted. Algorithm hyperparameters
remain in `config/ppo.json` and `config/sac.json` respectively.

All previous exploratory checkpoints and policy results were removed on
2026-09-05. New paper experiments start with fresh policies and a frozen
physical baseline. The retained calibration data precede policy deployment.

See [the paper protocol](../docs/PAPER_EXPERIMENT_PROTOCOL.md) for proposed
seed repetitions, common validation scenarios, required logging and figure
exports. Several reporting requirements, including complete validation history,
remain implementation work for the redesigned experiment workflow.
