# Where files belong

| Location | Use it for |
|---|---|
| `config/` | Active saved settings and model/flight catalogs |
| `data/raw_takes/` | Imported preliminary measurements |
| `data/flight_batches/` | Paired flight recordings, flown commands, protocols and timing |
| `runs/` | Fitted models, planning/training jobs, original predictions and audits |
| `exports/` | CSVs and bundles to hand to the flight program |
| `paper/figures/` | Figures and their provenance, ready for manuscript work |
| `third_party/natnet/` | Optional tracking SDK installed on this machine |
| `docs/setup/` | Installation, lab setup and transfer guides |
| `docs/methods/` | Model, identification and evaluation definitions |
| `docs/paper/` | Paper handoff, experiment protocol and literature |
| `docs/development/` | Existing development results and diagnostics |

Code remains in `simulator/`, `planning/`, `learning/`, `experimental_data/`,
`deployment/` and `tools/`. In particular, `experimental_data/` is Python code,
while `data/` contains measurements. Stable package names keep saved source
snapshots usable. The [run index](../runs/README.md) explains the run subfolders.

## Renamed locations

| Previous path | Current path |
|---|---|
| `rehearsal_csv_and_result_in_real_flight/` | `data/flight_batches/` |
| `policies/` | `exports/` |
| `output/pdf/` | `paper/figures/` |
| `.natnet_download/` | `third_party/natnet/` |

Batch names and all original data, protocols, models, forecasts, active configs
and frozen source snapshots retain their bytes and scientific meaning.
The research machine keeps local compatibility links for old evidence paths;
these links are not duplicate datasets and are ignored by Git. New discovery,
export defaults and guides use the new locations. Older checkout layouts remain
readable. Copy a portable baseline with the separate deployment branch/runbook
when transferring to another machine; historical absolute provenance paths
are not rewritten into claims of portable independent evidence.

The local `aerowhip` project path currently links to the original physical
checkout directory. The interrupted app-closing rename did not move that
directory. Do not attempt another app-closing migration during experiment work.
