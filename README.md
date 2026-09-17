# AeroWhip

Research implementation for aerial cable whipping: simulation, planning,
recorded-flight fitting, fixed-reference command correction, and evaluation.

## Entry points

- Research UI: `.venv/Scripts/python.exe run_simulation.py`.
- Lab UI: `.venv/Scripts/python.exe run_lab.py` (no job starts automatically).
- Command-correction options: `.venv/Scripts/python.exe tools/correct_reference.py --help`.
- Tests: see the [test guide](tests/README.md).

The research UI has five pages: Models & fitting, Recordings, MPPI, Rehearsals,
and Flight comparison. `run_simulation.py --headless` instead runs the historical
constant-force simulator; it is not a headless version of the PVA planning workflow.

Legacy SAC/PPO trainers and policy launch/replay controls have been retired.
The active workflow remains MPPI planning, recorded-flight fitting, fixed-reference
command correction, and PVA rehearsal/export. Existing experiment snapshots are
immutable; the pre-cleanup implementation remains in Git history.

## Models and experiment records

Use the [model catalog](config/evaluation/campaign.json) and the selected job's
saved configuration to identify models and artifacts. The catalog contains M0–M7
as of September 17, 2026; earlier M0-reset instructions describe a historical
collection, not the current workspace. A catalog entry is not flight approval.

Recordings, fitted assets, commands, and results remain in their existing locations.
Do not move these based only on directory names: saved jobs can reference them.
The current implementation and UI are retained; no replacement repository has
been created.

See the [folder map](docs/FOLDER_MAP.md), [documentation index](docs/README.md),
and [baseline preservation note](docs/development/REPOSITORY_BASELINE_20260917.md).
