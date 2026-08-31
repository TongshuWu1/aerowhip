# Legacy implementations

This directory preserves complete project baselines from before major research
direction changes.

- `retired_planning/` — superseded Milestone-4 MPPI and pre-production CEM
  runners/modules moved out of the active planner namespace.
- `retired_learning/` — retired SAC, deterministic amortization,
  diffusion/scorer, and residual-policy source/config/test snapshots.

Legacy contents are preserved for reference and selective reuse. They should
not be edited in place; make new work in the repository root and copy only the
components deliberately adopted by the new design.

The former multi-gigabyte full baseline snapshot is intentionally not tracked;
the original source remains recoverable from Git history.
