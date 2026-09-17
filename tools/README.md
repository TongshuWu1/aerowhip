# Current workflow commands

Run from the repository root with `.venv/Scripts/python.exe`.
Running a script is an explicit action; checking status never requires restarting
an existing fit, training job or planner.

| Task | Entry point |
|---|---|
| Desktop, five current pages | `../run_simulation.py` |
| Run an already prepared MPPI PVA job | `run_pva.py --job <job>` |
| Run a reviewed preliminary fit | `fit_preliminary.py --job <prepared-job>` |
| Compare frozen M0 forecast, prepare raw data, diagnose, fit justified damping | `adapt_whip.py <setup/compare/prepare/diagnose/fit> --help` |
| Register actual model lineage and compare models on a common recorded flight | `evaluate_models.py <seed/register/flight/evaluate> --help` |
| Produce a PVA rehearsal from a completed saved job | `rehearse_pva.py --help` |
| Parse preliminary native tracking/controller logs | `process_takes.py --help` |
| Inspect live MPPI snapshots | `open_mppi_live.py --help` |
| Benchmark current PVA execution | `benchmark_pva_performance.py`, `benchmark_pva_rollout.py` |
| Source-only review export | `build_source_release.py --output <new-folder>` |

The new ten-take workflow, generic fitting/correction commands, and initial-state
sensitivity diagnostic are documented in `../docs/FRESH_MODEL_ITERATIONS.md`.

The selected M0 flight package and forecast already exist. Do not recreate them
with current defaults. See `../docs/FRESH_MODEL_ITERATIONS.md` for staged commands and
the required clock/contact/data-role review. Fitting does not promote a model or
start a new plan. See `../docs/methods/SIM_REAL_EVALUATION.md` for M0/M1/M2 comparison.

SAC/PPO trainers, policy replay/export launchers, and the old overnight policy
campaign have been removed. Shared force-physics, CEM, recovery and recording
helpers remain where numerical checks or current workflows use them. The old five-take normalized fitter
is not the new whip adaptation workflow. Retired UI/one-off source can be located
through `../runs/audits/paper-readiness-cleanup-20260910/deletion_manifest.json`.

The source exporter contains code and unfitted configuration, not the learned
assets needed to reproduce the selected flight. It creates a local review
candidate; it does not publish or grant a license. Existing outputs are immutable.
