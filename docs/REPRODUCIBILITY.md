# Reproducibility and evidence

The development branch includes explicitly committed research data, models,
checkpoints, source snapshots, saved plans, CSVs and audits. Many numerical/binary
files use Git LFS: run `git lfs pull` after retrieving the branch. New files under
`runs/`, `data/` and `policies/` remain opt-in through existing ignore rules.

For each reported result retain the actual source/configuration, model and residual
hashes, seed, initial state, task/reward/termination definition, optimizer history,
command timestamps, complete recovery and validation selection procedure. For real
flights also retain raw measurements, masks, clock/frame mapping, sent commands,
vehicle identity and the exact original saved forecast.

Current example: [MPPI result record](MPPI_PULLBACK_20260909.md). Its final portable
CSV and all saved arrays regenerate exactly in the recorded environment. Its
continuation timing excludes the parent's optimization and cannot be reported as
a cold solve. The latest relevant suite passed 73 tests on Windows/RTX 4080;
see [paper handoff](PAPER_WRITING_HANDOFF.md) for the exact test list and metrics.

Distinguish simulation success, reference feasibility, modeled recovery, exact
software replay, prediction error against recorded motion, and prospective physical
performance. They are different outcomes. Current direct-PVA MPPI is simulation-only.
New PVA PPO has no trained matched performance baseline.

Split data by complete takes/sessions. Hover-normalized Z uses retrospective pre/post
data and must be disclosed. No renamed historical test becomes independent after
being used for development. Missing datasets/CUDA may produce explicit test skips;
report those separately from passes.

The legacy source-only release builder is a separate packaging path with its own
force-era assumptions; it is not a verified portable release of the current PVA
application. Use a saved PVA rehearsal package for the documented exact replay.
See [PUBLICATION.md](PUBLICATION.md) before preparing a separate public artifact.
