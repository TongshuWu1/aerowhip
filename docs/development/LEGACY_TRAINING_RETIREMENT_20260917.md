# Legacy trainer retirement

This follows the preservation merge `b5cc42a9a386c91497a489b0238b6ee9ee1d9ee1`.
It removes active SAC/PPO training and policy replay/export entry points, not
the experiment history. Deleted implementation and feature-specific tests are
recoverable from that Git revision.

## Scope

- MPPI no longer imports policy agents, policy rollout collectors or PPO reward adapters.
- The planner accepts MPPI only. Old policy configurations/checkpoint continuation
  are explicitly rejected, not silently interpreted as MPPI.
- Hidden PPO controls and policy-generation jobs were removed from the UI.
  The rehearsal inspector reads existing commands; generation stays on the MPPI page.
- Source/lab packaging no longer requires retired trainers or a PVA PPO configuration.
  Source packaging omits four workstation-specific manuscript-video scripts; their
  originals remain in the repository. Privacy scanning remains enabled.
- Mixed physics/export tests keep their numerical checks. Trainer-only tests are
  retired. The small historical force-environment configuration remains for shared
  force-backend regression tests; it is not a supported training entry point.

No saved model, recording, command bytes, objective weight, fitting tolerance,
frozen source snapshot, or paper figure is changed. One missing dependency,
`exports/M0_Bspline_slower_brake_1s/fullstate_30hz.csv`, is restored byte-for-byte
from the retained M0 rehearsal: its SHA-256 matches the frozen vertical reference.
It is required by current correction provenance, not a newly generated command.

The existing diagnostic-only-input rejection is moved before reading training
fields, fixing an early `KeyError`; no fitting algorithm or tolerance changes.
Two historical physics tests now use explicit synthetic fixtures instead of
assuming the unfitted UI template contains a fitted checkpoint.

The dated baseline JSON remains a
record of the earlier preservation pass, not a claim about the current source.

## Preservation checks

Compared with the preservation merge: the MPPI update function has identical
Python AST, and explicit MPPI defaults are identical. A 12-interval CPU replay
using the retained M0 physical/task settings and the jerk integrator matches
the old implementation exactly, including all result tensors and cable positions.
This is a regression check, not a physical-flight validation.

All 138 hashes in the eight active model-catalog entries verified after cleanup.
All nine command CSV hashes in the dated preservation record match, and all
seven frozen vertical-reference dependencies verify after restoring its CSV.

## Final verification

Command: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --tb=short`,
with `QT_QPA_PLATFORM=offscreen`, `PYVISTA_OFF_SCREEN=true`, and
`PYTHONDONTWRITEBYTECODE=1`.

Result: **779 passed, 29 skipped, 6 warnings in 537.42 seconds**. The warnings
are PyTorch JIT deprecations. Skipped checks are not claimed as validated.

This includes CUDA command-correction/export, physics and predictor parity,
fitting/data contracts, UI regressions, and a source-only package startup outside
the checkout. New tests reject retired methods/checkpoints and ensure active
pipeline imports do not load policy trainers. `git diff --check` is clean.

No new physical flight, fresh full-scale model fit, or new optimization campaign
was performed. The interactive screenshot audit helper was syntax-checked; the
verification above uses automated/offscreen UI tests, not manual visual review.
