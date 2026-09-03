# Aerial Cable Research Simulator

This repository contains the frozen aerial-cable simulator, the selected
closed-loop PPO whip controller, the stopped pure-SAC baseline, and the
production CEM reference planner.

## Current decision

The selected learned architecture is **10 Hz closed-loop PPO**:

```text
measured UAV + cable state
  -> normalized 83-D state/goal/physics context
  -> PPO query every 0.1 s
  -> acceleration + body-rate command
  -> full UAV/residual/12-node-DDER propagation
```

The selected D50 terminal PPO checkpoint achieved 480/512 = **93.75%**
deterministic success on the nominal physically propagated state-bank audit. Compiling the
same controller into an open-loop command reproduces nominal simulation
exactly, but takes about 12.17 s and loses substantial robustness under model
mismatch and post-start disturbances. Continuous PPO feedback is therefore the
current choice.

Production CEM remains the strongest offline reference: 98.05% first-seed and
98.44% up-to-three-seed success over 256 contexts, with 34.61 s median planning
time. Pure SAC is retained as a negative baseline: 106 successes in 1,206,272
episodes before it was stopped.

The concise evidence bundle is in [`results/`](results/README.md):

- [`results/ppo/`](results/ppo/README.md): checkpoints, plots, logs, compiler
  audit, and full PPO reports.
- [`results/sac/`](results/sac/README.md): stopped checkpoint, plots, and logs.
- [`results/cem/`](results/cem/README.md): benchmark rows, authoritative
  actions, plots, and report.
- [`results/common/`](results/common/): shared context normalizer and state
  banks used by current configurations.

See [PROJECT_GOALS_METHODS_AND_RESULTS.md](PROJECT_GOALS_METHODS_AND_RESULTS.md)
for the research goal, task definitions, method history, contributions, and
limitations.

## Frozen scientific model

- Freeze: `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`.
- CUDA float32, PCG32 cable damping.
- Attitude-coupled UAV model plus causal 100 ms learned residual.
- Twelve-node DDER cable, three DDER substeps, four position projections.
- Protected `fig8vertical_002` is not evaluated.
- No hardware execution is authorized by this repository state.

## Active entry points

For a new RTX 5090 workstation, follow
[`PORTABLE_WORKSTATION_SETUP.md`](PORTABLE_WORKSTATION_SETUP.md). PyCharm run
configurations are committed under `.run/`; after selecting the project `.venv`
interpreter, choose **01 Workstation Preflight** and press Run.

```powershell
# GUI and simulation inspection
.\.venv\Scripts\python.exe run_simulator.py

# The Simulator tab runs the frozen PPO checkpoint with one button.
# Headless equivalent:
.\.venv\Scripts\python.exe run_ppo_simulation.py

# Portable selected-policy continuation (expensive; plain Run performs preflight only)
.\.venv\Scripts\python.exe run_simple_ppo.py --train

# Retained pure-SAC baseline (normally do not resume)
.\.venv\Scripts\python.exe run_simple_sac.py --train

# Zero-training PPO feedback/open-loop compiler audit
.\.venv\Scripts\python.exe run_ppo_open_loop_compiler_audit.py

# Production CEM benchmark (expensive; curated result already exists)
.\.venv\Scripts\python.exe run_milestone6a.py

# Identification/refit workflow (expensive and not part of controller work)
.\.venv\Scripts\python.exe run_milestone3c.py

# Rebuild the curated results bundle from available historical run trees
.\.venv\Scripts\python.exe tools\curate_current_results.py
```

## Repository organization

- `simulator/`: coupled UAV, residual, DDER cable, and GUI.
- `fitting/`: model identification and validation.
- `planning/`: current CEM, rollout, action codec, metrics, and replay tools.
- `learning/`: current PPO/SAC control environments, networks, state/context
  interfaces, validation, and trajectory compiler.
- `config/`: current model, task, PPO, SAC, compiler, and CEM configs.
- `results/`: curated current evidence and checkpoints.
- `data/`: raw/processed measurements and active model assets only; historical
  policy/planning runs live in the dated sibling archive.
- `tests/`: regression coverage for retained code paths.

Historical milestone reports, retired learners, obsolete runners, and complete
run trees were moved—not deleted—to the dated sibling archive during repository
cleanup.
