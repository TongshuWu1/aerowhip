# Preserved implementation: September 17, 2026

This repository remains the existing working implementation. No new repository
was created. A future rewrite and UI redesign are outside this cleanup.

## Preservation

`archive/repo-cleanup-20260917-before/` contains a local pre-cleanup snapshot of
697 root/source/configuration/documentation/test files. SHA-256 comparisons
confirmed that all 697 matched the working files before documentation edits.
This is a source snapshot, **not** a complete backup of the repository, Git
history, dependencies, recordings, or fitted artifacts. Those remain in place.

The worktree already had 53 modified tracked files and 4,768 tracked deletions,
as well as untracked files, before this cleanup. These pre-existing changes were
not reverted or committed. Restore individual files from the snapshot only after
reviewing subsequent changes; do not restore it wholesale.

## Minimal cleanup scope

- Refresh the root README, documentation index, and folder map to remove stale
  M0-only workspace claims and identify the current entry points.
- Move generated `__pycache__` directories from the root and source/test packages
  to `archive/minimal-cleanup-20260917-caches/`, preserving their relative paths.
  Python can regenerate these caches; they contain no source implementation.
- Leave `.pytest_cache` unchanged because directory access is denied.
- Leave temporary experiment scripts, historical utilities, compatibility
  links, existing archives, and all experiment assets in their original locations.

No runtime code, UI, model, reward, fitting loss, convergence tolerance, command,
evaluation definition, or scientific result is changed by this cleanup.

## Known baseline checks

Before cleanup, a targeted run covering flight export, model evaluation, current
UI workflow, model-label renaming, and source-release packaging returned
20 passed and 2 failed. Both failures were in `tests/training/test_source_release.py`:
the builder expected a missing `config/pva/ppo.json` and a missing
`docs/development/M2_PPO_MPPI_MATCH.md`. Its UI smoke test also still expects six
pages rather than the current five. These packaging issues are not fixed here.

This baseline is not a claim that the full suite, numerical methods, or physical
flight safety have been validated.

After the documentation edits, the four non-packaging test files above were
rerun with offscreen Qt, bytecode writing disabled, and pytest's cache provider
disabled: **20 passed in 2.34 seconds**. The full suite was not run.
The cache archive contains 16 directories and 418 files; every moved file's
SHA-256 matched its original. No files were permanently deleted.
