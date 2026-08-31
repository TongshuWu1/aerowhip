# Active one-shot interfaces

No policy learner is currently active. This package now contains only the
stable interfaces shared by production planning and diagnostics:

- `policy_context.py`: root-centered, yaw-aligned 83-D context.
- `policy_action.py`: canonical normalized 49-D action and production codec.
- `one_shot_env.py`: open-loop batch evaluation helper.
- `state_bank.py` and `context_sampling.py`: physically propagated state and
  context construction.
- `cem_teacher_support.py`: read-only loading of verified Milestone-6A CEM
  teachers and the fixed production environment contract.
- `action_robustness.py`: deterministic perturbation banks used by the current
  teacher-manifold robustness audit.

SAC, deterministic imitation, diffusion/scorer, and residual-policy branches
are retired. Their source, configs, and tests are preserved under
`legacy/retired_learning/`; they are not imported by this package or the GUI.
Their immutable result artifacts remain under `data/policy_training/`.

The current diagnostic entry point is:

```powershell
.\.venv\Scripts\python.exe run_milestone7c.py --analyze-existing data/policy_training/cem_teacher_robustness_audit_v1/2026-08-30T213030.839170Z
```

That command only re-analyzes saved physics outcomes. It does not train a
policy or run CEM.
