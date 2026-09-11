# AeroWhip project instructions

Read `HANDOFF.md` first, then `docs/paper/PAPER_EXPERIMENT_PROTOCOL.md` for the active
study and `docs/paper/PAPER_WRITING_HANDOFF.md` for manuscript scope. Historical notes
do not override the user's latest decisions or establish live job status.

- Preserve the current selected models/policies, physical calibration, recent
  M0/M1/M2 work and their complete provenance: required preliminary inputs,
  original measurements, settings, flown CSVs, relevant failed trials, optimizer
  state, forecasts and immutable source snapshots. Never regenerate a forecast
  with a later model and present it as the original preflight prediction.
- Keep M0 and preliminary data unchanged. The new one-target study uses 5 M0 +
  5 M1 + 10 final paired executions. New update batches assign 001/002/004 to
  adaptation and 003/005 to operational validation. Do not change old roles.
  Final recordings stay outside fitting, stopping and tuning.
- Paper outcomes use continuous minimum 3D target distance, not binary 5 cm
  success. Existing planner/contact definitions retain their frozen semantics.
  Do not change objectives, model physics or selection rules during deployment
  or cleanup. Retained M0 lacks an active cable residual; full M1/M2 include it.
- Do not inspect, evaluate, plot, fit or tune against `fig8vertical_002`.
  Opaque copying for the authorized private lab transfer is permitted.
- Preserve the six-page main research UI and five-page deployment UI. The
  current PVA command/export rate is 30 Hz. Historical force/20 Hz checkpoints
  retain their own interpretation; do not convert them by relabeling.
- The simulator exports CSVs only. Offline planning/export is not flight
  authorization. No ROS flight sender is implemented and validated. Use actual
  vehicle/interface facts; do not invent frames, topics or firmware parameters.
- Keep tracked origin, attachment, center of mass and firmware frame distinct.
  Apply the saved rotated attachment offset consistently. Preserve native raw
  coordinates, missing-marker masks and clock-evidence limitations.
- Inspect current job status before touching PPO/MPPI jobs or supervisors.
  Do not duplicate or restart them automatically. Two-target work is paused;
  swing-and-settle is retired. Cleanup is not permission to start training,
  fitting, planning campaigns, model promotion or physical flights.
- Historical development comparisons are not independent final paper evidence.
  Keep conditional cable error, complete command-to-tip error and physical
  target distance distinct. Do not claim monotonic improvement from a mean.
- The user explicitly authorized deleting obsolete measurements, failed legacy
  runs, old records and superseded documents, while retaining the past two to
  three days' work, especially the 10 September M0/M1/M2 lineage. Check retained
  model/result dependencies before removing an old artifact. Preserve current
  selected PPO/MPPI and deployment work. Do not reset the working tree or
  rewrite Git history. This cleanup authorization does not authorize deleting
  unrelated external archives or bypassing an approval rejection.
- Prefer targeted existing tests. Record what actually ran and on which OS/GPU.
  Windows/RTX 4080 checks do not establish Ubuntu/RTX 5080 validation. Do not
  claim algorithmic or physical correctness from packaging/UI tests.

Keep these instructions concise. Completed-job snapshots can retain older
instructions for provenance; they do not override this active file.
