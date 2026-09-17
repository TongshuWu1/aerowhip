# Repository map

| Location | Purpose |
|---|---|
| `simulator/` | Simulation and research UI |
| `planning/` | Planning, fixed-reference correction, and job execution |
| `experimental_data/` | Recording import, fitting, and model evaluation |
| `deployment/` | Lab UI, rehearsal, and flight export |
| `learning/` | Shared PVA environment and learning implementations |
| `tools/` | Workflow commands and historical experiment utilities |
| `tests/` | Regression and integration checks |
| `config/` | Configuration, UI selections, and model catalog |
| `runs/` | Saved planning, fitting, correction, rehearsal, and audit artifacts |
| `data/` | Recordings and dataset metadata |
| `exports/` | Exported commands and associated flight recordings |
| `paper/`, `docs/` | Manuscript assets and documentation |
| `output/`, `tmp/` | Generated reports, working artifacts, and experiment scripts; not necessarily disposable |
| `workspace/`, `experiments/` | Local lab/runtime workspaces |
| `requirements/`, `third_party/` | Dependency specifications and external components |
| `archive/`, `delete/` | Recoverable local snapshots and earlier archives |

The model catalog is `config/evaluation/campaign.json`. It contains M0–M7 as of
September 17, 2026; this workspace is no longer an M0-only reset. Use catalog
entries and saved job metadata rather than a hard-coded model or export path.

Historical jobs can depend on files elsewhere in this tree, including old-looking
directories. The minimal cleanup did not relocate source, recordings, fitted
models, commands, or results. See the [preservation note](development/REPOSITORY_BASELINE_20260917.md).
