> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Research repository snapshot — 9 September 2026

The user requested committing and pushing the complete current project before
starting the proposed independent PhysX experiment. This snapshot includes the
source, configuration, documentation, tests, recorded flights, normalization
provenance, fitted models and residuals, retained PPO checkpoints, saved rehearsals,
policy packages, historical research results, and model-isolation audits currently
present in this repository.

The latest adapted PPO is
`runs/ppo/20260909-025902-328885-seed655/`. Its `checkpoints/` directory includes
`best_validation.pt`, `best_reward.pt`, `latest.pt`, and `terminal.pt`, alongside
the run's frozen configuration, source, and model assets. The selected
`best_validation.pt` SHA256 is
`424b5be3c20b8ba1b7d067eb4f53ee1a10045a860ff832d7853c02b565dd7fbb`.
Its fitted model is `data/model_candidates/20260908-normalized-M1/model.json`
(SHA256 `51d11f38bf8727f52c3641cbdce29f01cb1715c9313752a51299bfa43465e760`).
The original M0 flight policy remains under
`runs/ppo/20260908-195207-486249-seed655/`, with selected checkpoint SHA256
`d10657f471deb8b22cafbee9008792e8378c8c764ca97039d6dbe7afcc260eea`.
These saved runs remain stopped.

The latest research decisions and limitations remain in `../HANDOFF.md`. Saving
this snapshot does not restart training, refit a model, validate paused MPPI work,
or establish that M1 improved real-flight performance.

Git LFS stores checkpoints, numerical arrays, ZIP packages, and research CSVs.
Install Git LFS and run `git lfs pull` after cloning to obtain their contents.
Research snapshot files and CSVs disable Git newline conversion so that saved
hashes and the exact flown command bytes remain reproducible.

Machine-local virtual environments, IDE state, Python/test caches, the downloaded
NatNet SDK cache, and regenerable `live_scene` visualization feeds remain local.
Previously retired artifacts in archive directories outside the repository are
not restored into this snapshot. Raw recordings already in the repository are
preserved alongside their normalized derivatives.

Existing broad ignore rules continue to make newly generated research artifacts
opt-in; the present files were explicitly staged for this authorized snapshot.
