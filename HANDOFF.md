# AeroWhip: current handoff

## Initial hover clarification

The operator clarified that battery condition and damage are not considered the explanation, and reports small cable-tip motion during the initial hover, attributed to propeller airflow. The command-correction code initializes every rollout with the same frozen M0 cable shape and zero nodal velocity, rather than the measured pre-strike state of each take. Variation in the actual initial cable state is therefore a plausible source of tracking variability. Pre-strike motion and its association with strike error have not yet been quantified; neither airflow causation nor its contribution to the M1-to-M2 difference is established.
All five takes and measured metrics are unchanged. See
`runs/data_review/M2-paper-20260913/CONDITIONS_NOTE.md`.

## Latest physical check: five M2-selected flights

All five M2 takes were confirmed clean and matched the selected CSV. Tip and
marker coverage is complete over the strike window. The batch is evaluation-only;
no new fit, command correction or M3 training split was requested.

Equal-take means (M0 / M1 / M2): fixed-time target error 24.12 / 11.18 / 13.10 cm;
closest distance 14.73 / 7.83 / 8.60 cm; original tip-reference RMSE
18.09 / 16.06 / 13.28 cm; original quadrotor-reference RMSE
15.66 / 11.17 / 12.63 cm. Tip speed at original strike 5.98 / 5.85 / 5.82 m/s.
M2 improved average tip-trajectory tracking but not target accuracy over M1.
Fixed-time target-error SD increased from 4.04 to 7.52 cm. No take was excluded.
Clock offsets are estimated; M2_002 has 9.46 ms half-record disagreement.

Report: `runs/data_review/M2-paper-20260913/REPORT.md`.
Immutable intake: `data/flight_batches/M2_paper_selected_correction_20260913`.
Original files remain in `exports/M2_selected_fixed_tip_reference/flight_take`.
All raw and frozen-forecast hashes verified; no numerical source changed.

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
The five M2-selected flights are now evaluated above; the earlier values describe M0/M1.

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
not future flight performance. If a later fit is explicitly requested, its parent must be M2-selected; no fit is currently authorized.

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
