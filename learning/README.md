# Current learning package

This package contains the retained learned-control paths and shared production
context/action interfaces:

- `sequential_sac_env.py`: full-model sequential whip environment used by both
  PPO and the stopped SAC baseline.
- `simple_ppo.py`: selected PPO actor/value implementation.
- `ppo_validation.py`: deterministic state-bank validation and plots.
- `ppo_trajectory_compiler.py`: zero-training feedback/open-loop audit tools.
- `simple_sac.py`: retained pure-SAC negative baseline.
- `normalization.py`, `policy_context.py`, `policy_action.py`, `state_bank.py`,
  and `context_sampling.py`: shared production data contracts used by PPO/CEM.

Figure-eight SAC, one-shot SAC helpers, diffusion/amortization, flick, and
iterative-residual branches are historical and are not part of the active
package. Their source and full artifacts are recoverable from the dated sibling
archive.

Current evidence and checkpoints are organized under [`results/`](../results/README.md).
