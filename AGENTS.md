# Project instructions

Read [HANDOFF.md](HANDOFF.md) first. For paper writing, read
[docs/PAPER_WRITING_HANDOFF.md](docs/PAPER_WRITING_HANDOFF.md).

## Current scope

- Latest MPPI wave work is complete: `20260909-160208-467697` accepts an independently
  replayed full proposal from parent `20260909-155433-585040`. Parent is STOPPED
  with 13 committed actions; do not call its rolling loop completed. Derived plan
  has 41 actions, a modeled hit and full recovery/export. No worker remains.
  Fitting, PPO, campaigns and both heartbeats stay stopped/paused. No flight is authorized.
- Active commands are desired tracked-origin P/V/A at 30 Hz from bounded XYZ
  jerk. Historical force checkpoints and CSVs keep their original semantics.
- The current UI has six pages: Models & fitting, Recordings, PPO, MPPI,
  Rehearsals, Flight comparison. Do not restore obsolete CEM/SAC navigation.
- PPO and MPPI settings are independent (`config/pva/ppo.json`, `mppi.json`).
  The current PVA PPO and MPPI tasks differ; no matched performance claim exists.
- Latest accepted MPPI wave plan uses historical normalized M1, not unfinished M0.
  Three persistent dominant-bend stages must precede contact; this is a kinematic
  proxy, not proof of energy transfer. Full CSV/all-array portable replay is exact.
  Rolling lookahead is 2 s, 1,024 samples, one committed 30 Hz action per replan;
  the separate 5 s maneuver limit is not a planning-time cap.
- The intended strike requires forward aircraft pull then backward release
  while the tip continues forward. The current task checks actual modeled
  aircraft reversal at interpolated contact, not just command reversal.
  Current MPPI additionally checks travelling-bend progression. PPO reward/task
  configuration remains independent and unchanged; do not force identical shaping.
- Start `[-2,0,1.255]` m and target `[-1,0,1.1]` m match PPO central coordinates.
  Saved historical outputs must not be translated/relabelled to new defaults.

## Preserve research meaning and artifacts

- Preserve raw logs, failed trials, masks, models, checkpoints, optimizer state,
  source snapshots, flown CSV bytes and their exact saved prediction ghosts.
  Do not regenerate an old ghost with a newer model and call it the original forecast.
- Measured geometry is tracked-origin/attachment-specific. Apply the rotated
  offset consistently; never assume tracked origin equals COM or firmware origin.
- Hover-normalized Z uses retrospective pre/post information. State that limitation
  in paper claims; it is not physical flight validation or measured coordinate calibration.
- Keep cf7 and cf3 data/model identities separate. M0/M1 names require exact provenance.
  All-data development fits are not independent tests. Preserve old split snapshots.
- No change to a provisional simulation bound establishes a measured vehicle limit.
  No verified ROS flight sender was added by the PVA implementation. Use actual
  firmware/interface details rather than inventing topics, frames or parameters.
- Latest user clarification: the old sudden drop was caused by acceleration demands
  exceeding drone capability. Battery effects on other tracking errors remain unconfirmed.
  Bolt battery is 2S; PVA-based battery calibration is proposed, not implemented.
- Read `docs/FUTURE_ADAPTATION_FITTING.md` before any newly authorized fit.
  Use practical plateau stopping, preserve best weights/state, and distinguish
  convergence from manual stops or safety ceilings. No automatic fold/ablation campaigns.
- Historical selected flight policy: `runs/ppo/20260908-195207-486249-seed655/checkpoints/best_validation.pt`,
  SHA256 `d10657f471deb8b22cafbee9008792e8378c8c764ca97039d6dbe7afcc260eea`.
  Later separate study outputs do not rewrite which policy produced earlier flights.
- Preserve independent legacy CEM/force/Isaac assets. PhysX work stays paused.
- Do not reset the working tree, rewrite Git history, or permanently remove
  previously protected research artifacts as cleanup. Retired external archives
  remain recoverable through their manifests; do not recreate or delete them.
- Prefer targeted existing tests. Report the actual tested hardware and OS;
  current PVA verification is Windows/RTX 4080, not Ubuntu/other-GPU validation.

## Documentation

`docs/README.md` is the active reading list. Superseded test reports, run diaries,
old UI instructions and prior versions of these instructions are under
`docs/history/20260909-doc-cleanup/`. They describe historical decisions and
must not be treated as live run instructions. Keep necessary evidence accessible
through links rather than restoring obsolete progress notes to the active index.
