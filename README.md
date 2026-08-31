# Aerial Cable Research Simulator

This repository contains one frozen aerial-cable simulator, one production
open-loop CEM reference planner, and an experimental sequential PPO whip
learner. There is no promoted or hardware-ready neural policy.

## Current scientific status

- Model: `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`.
- Production predictor: attitude-coupled UAV physics + causal 100-ms UAV
  residual + 12-node DDER cable.
- Planner: variable-duration production CEM over one normalized 49-D action.
- Benchmark: 251/256 first-seed and 252/256 with at most three seeds; zero
  population-to-authoritative replay flips.
- Important limitation: the saved successful actions are locally knife-edge.
  At normalized acceleration noise sigma 0.005, scientific-success survival is
  20.87% for IID noise and 10.31% for smooth noise.
- Direct policy learning history: SAC, deterministic regression, and
  diffusion/scorer did not establish a deployment policy. A from-scratch
  sequential PPO run has now completed 1,001,472 episodes with 9.99%
  cumulative success, 74.14% success over the final 10,240 episodes, and 10/10
  deterministic successes on a fixed mildly varied validation set. This is a
  canonical-target simulation result, not yet a general or hardware-ready
  policy.
- The Figure-8 SAC experiment was stopped and is not an active path.
- Iterative residual pilot: using only existing perturbation data, learned
  correction increased state-disjoint development success from 16.22% to
  43.24% after at most five corrections. This is partial evidence for
  outcome-guided refinement, not a one-query or hardware-ready controller.
- Targeted continuation: 22,528 additional simulator responses were collected
  around failed initializer actions with trajectory features. Training success
  increased, but state-disjoint development fell to 37.84%; that checkpoint is
  not promoted. Joint/edge contexts showed very sparse local success support.

The selected residual checkpoint remains the earlier 43.24% model. More
gradient steps or nominal CEM winners are not justified. Any next method must
first address sparse correction support for joint/edge state-target contexts.

See [PROJECT_GOALS_METHODS_AND_RESULTS.md](PROJECT_GOALS_METHODS_AND_RESULTS.md)
for the complete project goal, method history, negative results, contributions,
and the final PPO reward/result.

## Active pipeline

```text
physical takes
  -> deterministic preprocessing / PhysicalEpisode
  -> decomposed identification (UAV physics, EI/Cb, causal residual)
  -> explicitly selected model freeze

physically propagated initial state + target + nominal theta
  -> root-centered, yaw-aligned 83-D PolicyContext
  -> production CEM proposes normalized complete action [49]
  -> one production decoder
  -> 16 acceleration knots + T_maneuver in [0.45, 1.80] s
  -> ACTIVE command
  -> 0.30-s analytic smooth SETTLE
  -> stationary HOLD to T_evaluation = 2.40 s
  -> fixed numerical batch contract 2048
  -> UAV physics + residual + 12-node DDER rollout
  -> unchanged scientific hard gates
  -> authoritative top-32 replay and final selection

saved verified CEM actions
  -> deterministic perturbation banks
  -> same production decoder and authoritative simulator
  -> local success-survival / hard-gate-margin audit

saved perturbation actions + observed physical outcomes
  -> learned delta-outcome model
  -> compact 13-D correction proposals around a full 49-D action
  -> neural hard-gate ranking (no simulator/CEM candidate search)
  -> execute one correction and observe again, up to five iterations
```

There is no online CEM, scorer, replay buffer, or hardware execution in the
GUI. The Planning page is a read-only saved-result viewer. The Training page
can launch, cooperatively stop/resume, and plot the isolated PPO experiment;
it does not expose a policy for hardware execution.

## Active entry points

```powershell
# GUI: simulator, data, identification, planning replays, and PPO monitoring
.\.venv\Scripts\python.exe run_simulator.py

# Current directional task-whip PPO (also launchable from the Training page)
.\.venv\Scripts\python.exe run_simple_ppo.py --train --config config/learning/whip_ppo_once_directional_d15_v1.json

# Unattended balanced/compact/strike reward comparison + final audit
.\.venv\Scripts\python.exe run_whip_ppo_reward_study.py

# Identification/refit workflow (expensive; do not run for a GUI check)
.\.venv\Scripts\python.exe run_milestone3c.py

# Production CEM benchmark (expensive; saved result already exists)
.\.venv\Scripts\python.exe run_milestone6a.py

# Current saved-teacher robustness diagnostic
.\.venv\Scripts\python.exe run_milestone7c.py --analyze-existing data/policy_training/cem_teacher_robustness_audit_v1/2026-08-30T213030.839170Z

# Existing-data iterative residual pilot (no new CEM)
.\.venv\Scripts\python.exe run_iterative_residual_policy.py

# Separate simple Figure-8 SAC (one-million-episode config)
.\.venv\Scripts\python.exe run_figure8_sac.py
```

## Production numerical contract

- CUDA float32.
- PCG32 cable damping backend.
- Three DDER substeps and four position projections.
- Fixed UAV/residual numerical evaluation shape 2048.
- Normalized action `[49]`: 16x3 acceleration knots plus maneuver duration.
- Scientific gates: tip error <= 50 mm, directed speed >= 4 m/s, direction
  error <= 30 degrees, c10 first, UAV displacement <= 0.50 m, UAV speed <=
  3 m/s, command acceleration <= 20 m/s^2, finite full rollout.

## Repository boundary

- `simulator/`: active coupled simulator and GUI.
- `fitting/`: active identification and validation code.
- `planning/`: active production command, rollout, metric, selection,
  model-freeze, result, and replay contracts.
- `learning/`: stable context/action interfaces, robustness diagnostics, the
  stopped simple-SAC baseline, and the IRP-style delta-outcome diagnostic.
- `data/`: immutable raw/processed evidence, model freezes, planning outputs,
  and historical policy artifacts.
- `legacy/retired_learning/`: source snapshots of retired policy branches.
- `legacy/retired_planning/`: superseded MPPI and pre-production CEM code.
- `reports/archive/`: historical milestone reports.

The root retains only current model/planner/diagnostic reports. See
`PROJECT_ACTIVE_PIPELINE_AND_ZOMBIE_CLEANUP_REPORT.md` for the complete audit.

`fig8vertical_002` remains protected and was not evaluated. Real hardware was
not executed.
