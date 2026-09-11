# Reproducibility and evidence

The development branch includes explicitly committed research data, models,
checkpoints, source snapshots, saved plans, CSVs and audits. Many numerical/binary
files use Git LFS: run `git lfs pull` after retrieving the branch. New files under
`runs/`, `data/` and `exports/` remain opt-in through existing ignore rules.

For each reported result retain the actual source/configuration, model and residual
hashes, seed, initial state, task/reward/termination definition, optimizer history,
command timestamps, complete recovery and validation selection procedure. For real
flights also retain raw measurements, masks, clock/frame mapping, sent commands,
vehicle identity and the exact original saved forecast.

Current example: the selected package `runs/flight_packages/20260910-022818-648386`.
Its command CSV and original forecast are frozen and checksummed. See
[paper handoff](../paper/PAPER_WRITING_HANDOFF.md) for exact identities and evidence limits.
The current code/release audit is `runs/audits/paper-readiness-cleanup-20260910`;
its reported tests must be distinguished from experimental validation.

Distinguish simulation success, reference feasibility, modeled recovery, exact
software replay, prediction error against recorded motion, and prospective physical
performance. They are different outcomes. Current direct-PVA MPPI is simulation-only.
New PVA PPO has no trained matched performance baseline.

Split data by complete takes/sessions. Hover-normalized Z uses retrospective pre/post
data and must be disclosed. No renamed historical test becomes independent after
being used for development. Missing datasets/CUDA may produce explicit test skips;
report those separately from passes.

The source-only builder includes current PVA code with no selected learned assets.
An isolated-directory UI smoke test checks imports and empty-state behavior using
the existing interpreter; this is not a fresh installation test. Use the frozen
experiment package and source snapshot for the selected result's exact provenance.
See [PUBLICATION.md](PUBLICATION.md) before preparing a separate public artifact.
