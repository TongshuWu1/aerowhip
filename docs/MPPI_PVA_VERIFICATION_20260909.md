# MPPI PVA verification — 9 September 2026

MPPI planning, complete recovery, portable export and the Windows UI pass the
current checks. Fitting and PPO remain stopped; the user is investigating the
physical drone tracking/height problem. No flight was performed.

## Model and task

These are simulation diagnostics using frozen historical normalized M1,
`data/model_candidates/20260908-normalized-M1/model.json`, SHA256
`51d11f38bf8727f52c3641cbdce29f01cb1715c9313752a51299bfa43465e760`.
The unfinished fresh M0 was not published or reused as a completed fit.

Unchanged task: tracked origin [0,0,1.225] m, target [1.5,0,1.1] m, two-second
maximum whip, 30 Hz bounded XYZ jerk, 5 cm tip-first contact radius, at least
4 m/s directed tip speed, maximum strike angle45°. Same fitted drone/cable
equations and residuals, rewards, jerk bounds and PVA/workspace constraints.

## Corrections

- Saved best reward, distance, failure and hit status now describe the same
  retained action sequence. An optimizer completion is distinct from a hit.
- Atomic NPZ publication uses the existing Windows reader-lock retry helper.
  Rehearsal is blocked during active updates; duplicate worker launches cannot
  overwrite an already completed status. Cooperative stops retain saved plans.
- MPPI evaluates the current mean as one extra trajectory. This deterministic
  candidate is excluded from Gaussian importance weights; the correlated
  change-of-measure formula is unchanged. There are1024 random candidates plus
  the mean in parallel. No PPO policy or CEM elite update is used.
- A256-candidate-per-scale diagnostic found feasible counts163,101,24,8,0
  for latent noise0.02,0.05,0.1,0.2,0.5. The original default run produced only
  failed candidates and correctly refused export. Active MPPI noise is now0.05
  instead of0.5; temperature1 instead of10 makes reward differences more
  influential. These are documented sampling changes, not changes to rewards.
- Active PVA recovery uses the selected run's saved acceleration/speed/tilt
  envelope. The legacy force recovery's extra5 m/s² horizontal shaping limit
  rejected this otherwise recoverable fast exit. The new moving return reaches
  about11.08 m/s² horizontal acceleration while respecting the saved PVA
  specific-force, tilt, speed and height constraints. Historical force recovery
  defaults and artifacts remain unchanged; the saved whip was not edited.
- Export rejects incomplete recovery predictions and predicted drone/cable
  height violations. Metadata includes commanded/predicted height ranges.
- UI shows the saved plan's validity, prevents live rehearsal, keeps unavailable
  models disabled and lists rehearsals by actual metadata recency rather than
  arbitrary folder names. Old smoke folders no longer precede the latest run.

## Verified result

Optimization `runs/mppi_pva/20260909-114422-355245` stopped on its reward plateau
after27 iterations in141.73 seconds on Windows, RTX4080, PyTorch2.11.0+cu128.
The selected plan first achieved a modeled valid hit at iteration2; the final
retained strike has reward299.2512 and minimum tip distance0.0309370 m.

The final selectable replay job is
`runs/mppi_pva/20260909-115040-750483`. It freezes the updated recovery/export
code around the original plan; the optimizer was NOT rerun. Its source history
and parent are explicit in identity.json. Both plan files have SHA256
`2d45a53c44dbcaf7c423f874356290cb6d6241d591eb1913edacc2f19489473c`.
Earlier failed/intermediate diagnostics are archived from the UI, with all
files retained. No historical policy or flight artifact was modified.

Final rehearsal:
`runs/rehearsals_pva/20260909-115040-750483-mppi-diagnostic`.

| Check | Result |
|---|---|
| Modeled hit time |1.45437 s|
| Unchanged whip command ends |1.46667 s|
| Complete command and prediction |11.73333 s, including return and final hold|
| Commanded height range |1.15713–2.59542 m|
| Predicted drone height range |1.15106–2.48149 m|
| Minimum predicted cable height |0.14563 m|
| Saved drone height limits |0.96–2.8 m|
| Planner/export drone prefix difference |1.24e-14 m|
| Planner/export cable prefix difference |3.06e-13 m|
| Portable CSV regeneration |Byte-identical|
| Portable saved-array regeneration |Exactly zero difference in every array|

Portable package:
`runs/audits/mppi-pva-final/MPPI-PVA-simulation-diagnostic.zip`.
CSV SHA256:
`f2ee46e31ec43f430cd3ff9b2d75a91f07a17c779757dc27ad1bcf1c8be296be`.

## Evidence and limits

43 targeted tests passed in13.22 seconds, covering CUDA parity, PVA contracts,
optimizer weighting/retention, duplicate launch/stop behavior, recovery/export
and UI state. Native Windows Qt/VTK rendered all six pages and the final saved
MPPI rehearsal without errors. The11 original normalization source hashes
still match. Fit/campaign status remains stopped; heartbeat remains PAUSED;
the final process inspection found no Python worker running.

Detailed evidence: `runs/audits/mppi-pva-final/result.json`, `verification.json`
and `ui/`; earlier sampling and rejected-run evidence remains in
`runs/audits/mppi-pva-continuation` and `mppi-pva-refined`.

This verifies software and one selected simulated trajectory. Importance
weights often concentrate on approximately one sample and later sampled
batches have many failures; this is not a robustness or cross-seed success
claim. The historical model does not establish tracking on the current or
repaired drone. Recovery limits are provisional model constraints, not verified
actuator limits. No fit, PPO restart, physical flight or new flight evidence.
