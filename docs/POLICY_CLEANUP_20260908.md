# Current policy library and archived history

The user requested removing all learned-policy history except the policy currently
being worked on. The active project now retains one PPO run:
`runs/ppo/20260908-192755-046598-seed655`, named
**M0 flight PPO - lower setup - start X -2 target X -1**.

Retained its best-validation, best-reward, latest and terminal checkpoints,
training records, model/residual assets and frozen source. They are checkpoints
of the same working run. The current source checkpoint and parent configs were
also copied into its `provenance/` folder before archiving the parent, so lineage
can be inspected without restoring an old run into the library. Historical
`resumed_from` and launch paths remain original provenance, not instructions to
restart an archived ancestor. Continue from the retained run's checkpoint.

The 20,480-attempt continuation has completed at 314,368 total attempts.
`best_validation.pt` was selected at 308,224, with 245/256 development successes
(95.703125%), mean reward 225.5921. This is a development result; the run ended at
its attempt budget, not established reward convergence. No new training started
during cleanup.

## Archive

Recoverable archive outside the project:
`C:/Users/wts28/Documents/PHD/particle_filter_cable_project_archive_20260908_policy_cleanup`

Moved 117 top-level entries: 20 old PPO runs, 73 older rehearsals, 13 old policy
exports/library entries (including their README), and 11 old presentation
artifacts. The SAC run folder was already empty. The archive contains 10,519
verified files, approximately 6.71 GiB. Permanent deletion was not performed.

The archive retains the original project-relative folder structure. Its
`manifest.json` maps each old source path to its archived destination and records
every file's SHA256. `moved_entries.json` records the completed moves. The old
policy deletion registry is preserved under `config/policy_library.json` there;
the active registry is cleared and points to the archive manifest.

Raw recordings, calibration datasets, adaptation rounds, model-fit bundles and
the `rehearsal_csv_and_result_in_real_flight` folder remain in place. Historical
records/docs may refer to archived policy paths; use the manifest to locate them.
Research audits are retained. Archive contents are outside automatic policy and
rehearsal discovery.

## Current export and checks

One fresh rehearsal was generated from the retained best checkpoint at the
actual lower setup, without translating outputs:
`runs/rehearsals/20260908-193608-967519-current-policy`.

- Start tracking origin `[-2, 0.01287427254333901, 1.255]` m; target `[-1,0,1.1]` m.
- Predicted valid hit at 0.9466667 s; complete feasible recovery prediction.
- Commanded peak 2.62075 m; predicted drone peak 2.38694 m; cable peak 2.61336 m.
- Current portable bundle: `policies/Current-PPO-30Hz.zip`.

All archived files were hash-verified after moving. All 273 pre-existing active
run files checked remained byte-identical. Latest checkpoint loaded with both
optimizers; model asset hashes verified. Headless Qt verification showed one
training run, four current checkpoints, correct default best-checkpoint selection,
and a working current-rehearsal load/export path. Translation controls are absent.
GPU rehearsal tested on Windows / RTX 4080. No real flight validation is implied.

Audit: `runs/audits/20260908-policy-cleanup/verification.json` and `archive_plan.json`.
