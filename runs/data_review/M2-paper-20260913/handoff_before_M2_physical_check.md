# AeroWhip: current handoff

## Current flight and model

Main branch; current model is **M2-selected**: updated nominal quadrotor response,
retained M1 quadrotor residual, M2 cable physical parameters and verified cable
residual checkpoint 100. The user chose to keep the drone residual. No further
fitting or optimization is running. The cable-only variant is historical.

- CSV: `exports/M2_selected_fixed_tip_reference/fullstate_30hz.csv`.
- Raw next takes: `exports/M2_selected_fixed_tip_reference/flight_take/`.
- 217 rows, 30 Hz, 7.2 s. Origin (0,0,1.4) m; target (1.25,0,1.25) m.
- CSV SHA256: `c9a39713d627ceb5fe3cc555ccb2d3fdf853d3e20ae1a9c583faa661fbf8f379`.
- Model: `runs/adaptation/M2-selected-20260913/candidate/model.json`.
- Correction: `runs/reference_tracking/M2-selected-local-20260913-042637-826368`.
- Rehearsal: `runs/rehearsals_pva/20260913-042637-826368-M2-selected-local-fixed-tip-reference`.
- `exports/CURRENT_FLIGHT.json` and `config/pva/replay.json` select this run.

The corrected quintic B-spline starts from the executed M1 commands and matches
the same original M0 predicted physical tip trajectory and timing. The objective
is mean squared tip-reference error + 0.1 quadrotor-reference error + 0.01 command
position change from original M0. No direct measured-trajectory bias is added.
The fixed reference is `runs/reference_tracking/M0-paper-fixed-reference`.
Original strike time is 1.1172482457473654 s; correction prefix ends at 34/30 s.
Same slower braking, return to launch and final hold. Commands remain desired
tracked-origin P/V/A, kinematic acceleration, zero yaw, 30 Hz zero-order hold.

## Physical evidence and fitting

M0 and M1 each have five clean physical takes. Mean target error at the original
strike time decreased from 24.12 to 11.18 cm. Mean closest distance decreased
from 14.73 to 7.83 cm. Original-reference tip RMSE decreased from 18.09 to 16.06 cm.
These are physical results; M2-selected has not flown yet.

- Current M0 data: `data/flight_batches/M0_paper_slower_brake_20260913`.
- Current M1 data: `data/flight_batches/M1_paper_local_correction_20260913`.
- Data checks: `runs/data_review/M0-paper-20260913` and `M1-paper-20260913`.
- M1: `runs/adaptation/M1-paper-20260913-verified-physical`.
- M2 original completed fit: `runs/adaptation/M2-paper-20260913` (dependency only).

M1_001/002/004 were fitted; M1_003/005 were later explicitly used for development
model selection. Do not call them independent final test evidence. Selected
model prediction RMSE on these recordings: quadrotor 6.72 cm, combined tip 8.46 cm.
The full retrained M2 drone residual regressed. Cable-only ablation gave 5.79 cm
quadrotor and 9.53 cm tip prediction RMSE. These are retrospective predictions,
not future flight performance. Next fitting parent must be M2-selected.

Cable residual fitting stopped under the user's time budget; checkpoint 101
failed the unchanged numerical check, checkpoint 100 passed. No indefinite
retraining. Preserve causal masked initialization and observed measurements;
do not invent missing marker samples or normalize retrospective heights.

## Cleanup and reproducibility

Obsolete artifacts were moved to `delete/cleanup-20260913/`, preserving relative
paths. `plan.json`, `file_manifest.json` and `move_verification.json` record the
moves and byte verification. No permanent deletion. The previous complete
handoff is `delete/cleanup-20260913/before_cleanup/HANDOFF.md`.

Required older planning seeds, preliminary data, fitting jobs and source
snapshots remain at their original paths because current evidence depends on
them. Their retention is explained by `plan.json` under retained_dependencies.
All source packages, tests, calibration, current measurements, model weights and
flown CSVs remain. Legacy catalog entries are preserved in the archived catalog.
Do not reactivate old planner/reward experiments or bulk-delete dependency jobs.

## Paper and validation

The active manuscript is in the user's Dropbox Overleaf folder:
`C:/Users/wts28/Lehigh University Dropbox/TonyLehigh Wu/Apps/Overleaf/AeroWhip-ICRA/main.tex`.
The repository manuscript is an older reference. See
`docs/paper/PAPER_WRITING_HANDOFF.md` and `docs/paper/PAPER_EXPERIMENT_PROTOCOL.md`.

Current CSV, B-spline derivatives, PVA joins, complete replay and offscreen UI
were checked on Windows / RTX 4080. These are not Ubuntu/5080 validation or
physical collision-clearance certification. There is no validated flight sender.
Do not inspect or evaluate the protected fig8vertical_002 recording. Preserve
raw data and frozen source hashes; cleanup is not authorization to refit or fly.
