# Direct PVA workflow

The current application plans offline in desired position, velocity and
acceleration at 30 Hz. Read [the paper handoff](../paper/PAPER_WRITING_HANDOFF.md) for exact
equations, selected model and experiment identity, and [HANDOFF](../../HANDOFF.md)
for live status. Earlier force-policy commands are not PVA jerk actions.

## Six-page desktop

Run `.venv/Scripts/python.exe run_simulation.py`.

| Page | Purpose |
|---|---|
| Models & fitting | Select an actual fitted model, monitor a reviewed preliminary job, inspect saved diagnostics |
| Recordings | Preliminary takes, flight batches, recording checklist |
| PPO | Independent policy setup, progress, checkpoint library and checkpoint rehearsal |
| MPPI | Independent trajectory setup, running statistics/logs, live candidate view and saved plans |
| Rehearsals | Inspect the exact saved command/forecast, scrub the 3D scene and export |
| Flight comparison | Model lineage, common-flight predictions, prospective outcomes and adaptation progress |

Opening a page does not fit, train or plan. New jobs own their configurations and
component assets. Changes to setup affect subsequent jobs; a saved forecast keeps
its original model, coordinates and success criterion. Archived runs are hidden
from the normal list without removing their evidence.

## Planning and rehearsal

Bounded XYZ jerk integrates into consistent PVA knots. The loaded-aircraft model
predicts tracked-origin pose from held packets; the rotated attachment drives
cable dynamics. The selected M0 uses scalar damping and no cable NN.

The selected MPPI-inspired implementation searches the complete maneuver in
control-point space, with GPU batches and editable baseline proposals. It is an
offline optimizer. Other saved runs may use the separate receding-search variant.
Lookahead, maneuver duration and wall-clock planning time are distinct.

Active MPPI success is feasible tip entry into the declared target sphere.
Direction, fold shape, forward preparation and backward release are preferences
or diagnostics. Their values do not veto `tip_contact_v1`. Historical jobs and
PPO retain original semantics; their scores/hit rates are not interchangeable.

Rehearsal appends modeled recovery and final hold, then retains exact command
CSV and prediction arrays. Partial live plans cannot be exported as complete
maneuvers. Use the selected package README for the current flight. The real
sender/controller interface must be verified separately; the repository does
not supply a verified ROS flight sender.

## Real data and adaptation

Record the exact sent command log and native tracking. Preserve raw files,
marker gaps, frame conventions and clock evidence. First compare the flight
against its original frozen forecast. Then diagnose errors using identical
causal initialization and explicit observation masks.

Use [M0→M1 adaptation](../../delete/cleanup-20260913/docs/development/M0_TO_M1_ADAPTATION.md) for the reviewed raw-data steps and
[evaluation protocol](../methods/SIM_REAL_EVALUATION.md) for M0/M1/M2 comparisons. The model
library and retrospective fit loss are not evidence of prospective improvement.
Candidate fitting, registration, selection and a subsequent physical test are
separate actions. The current inbox is empty and only real M0 is registered.

Detailed earlier implementation/run notes are preserved in the pre-cleanup
source archive identified by `runs/audits/paper-readiness-cleanup-20260910`.
They are historical, not instructions to restart old campaigns.
