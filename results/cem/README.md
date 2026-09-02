# CEM results

This is the stabilized production variable-duration CEM benchmark using the
normalized 49-D action, ACTIVE → 0.30 s smooth SETTLE → HOLD command semantics,
and authoritative fixed-2048 numerical contract.

- Canonical: 6/6 seeds passed authoritative replay.
- 256 contexts: **98.05%** first-seed success and **98.44%** success with up to
  three deterministic seeds.
- Population PASS → authoritative FAIL: 0.
- Planning latency: 34.61 s median and 105.46 s p95.

CEM is retained as the strongest offline planner and scientific reference, but
its latency makes it unsuitable for immediate state-conditioned deployment.

Contents:

- `plots/`: benchmark success by context group and planning-time distribution.
- `data/`: context manifest, row-level benchmark data, authoritative actions,
  codec/replay audits, and summaries.
- `reference/`: the minimal historical canonical action used to seed a rerun.
- `replays/`: the 2.40-s production variable-duration canonical CEM replay
  shown by the GUI, plus the retained 0.70-s historical MPPI replay.
- `reports/`: the complete production CEM report.
