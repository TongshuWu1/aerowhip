# Current adaptation policy

`PPO-adaptation-20260908-195207.zip` contains the user-selected PPO,
its saved drone/cable models and both residuals, and the matching complete
30 Hz FullState CSV and rehearsal.

- Policy: `20260908-195207-486249-seed655 / best_validation.pt`.
- Policy SHA256: `d10657f471deb8b22cafbee9008792e8378c8c764ca97039d6dbe7afcc260eea`.
- Saved rehearsal: `runs/rehearsals/20260908-203914-039721`.
- Tracked-origin start: `[-2, 0, 1.255]` m; target: `[-1, 0, 1.1]` m.
- CSV SHA256: `d54b86aa6a1584a56ce1a063d29e47cb1ca91ea89c553e818ac33505e7225452`.

This package copies the saved trajectory exactly; no trajectory was regenerated
or translated during cleanup. The matching rehearsal is a simulation record.
For adaptation, identify which exact CSV was executed with each real recording.

The CEM ZIPs are independent planner results and remain unchanged.
Superseded PPO artifacts are recoverably archived at
`C:/Users/wts28/Documents/PHD/particle_filter_cable_project_archive_20260908_adaptation_policy_195207`.
Its `manifest.json` records original paths and hashes. Earlier archives and all
experiment/calibration/adaptation data are preserved.
