# Where files belong

| Location | Purpose |
|---|---|
| `exports/M2_selected_fixed_tip_reference/` | Current next-flight CSV and new raw takes |
| `exports/M0_Bspline_slower_brake_1s/` | Original flown M0 command and recordings |
| `exports/M1_local_fixed_tip_reference/` | Original flown M1 command and recordings |
| `data/flight_batches/` | Retained preliminary, M0 and M1 original measurements |
| `data/raw_takes/` | Imported preliminary measurements |
| `runs/adaptation/` | Active model lineage and required fitting dependencies |
| `runs/reference_tracking/` | Fixed M0 reference and command corrections |
| `runs/rehearsals_pva/` | Original commands and predicted motion |
| `runs/data_review/`, `runs/evaluation/` | Current data checks and prediction evidence |
| `config/` | Settings, calibration and active model/flight selection |
| `paper/figures/` | Retained recent figures and provenance |
| `docs/` | Current guides and technical references |
| `delete/cleanup-20260913/` | Reversible holding area for obsolete files; no permanent deletion |

Python packages, entry points, tests and dependencies retain their original
locations. Some older runs remain because current models or frozen forecasts
reference them. The cleanup plan lists these dependencies.

Compatibility links remain: `rehearsal_csv_and_result_in_real_flight` points to
`data/flight_batches`, `policies` to `exports`, `output/pdf` to `paper/figures`,
and `.natnet_download` to `third_party/natnet`. They are not duplicate datasets.

The live manuscript is in Dropbox Overleaf, outside this repository. See the
[paper handoff](paper/PAPER_WRITING_HANDOFF.md). The repository's manuscript is
historical; cleanup did not edit or move the Dropbox source.
