# Deeper cleanup review — September 17, 2026

Status: **completed after renewed user confirmation**. The initial combined
move/delete operation was blocked before it started. After the user confirmed
the cleanup again, a scoped file-only move succeeded: all 287 files were removed
from their active locations and their recovery copies were SHA-256 verified.
No recursive directory deletion was performed. Scoped LaTeX build-metadata
ignore rules were also added to `.gitignore`.

## Reviewed removal candidates

The local manifest is `archive/deep-cleanup-20260917/manifest.json`. It records
original relative paths, byte lengths, and SHA-256 hashes for 287 files totaling
69,317,733 bytes (about 66.1 MiB):

- 89 files in 14 old manuscript compilation/preview directories under
  `paper/figures/`.
- Five earlier compiled revision/draft/supplementary PDFs in `paper/figures/`.
- One root-level pre-edit manuscript backup, already included in the earlier
  source snapshot.
- 192 `main.aux`, `main.blg`, `main.out`, and `main.log` files in temporary
  compilation directories that also contain `main.pdf`.

The recovery destination is `archive/deep-cleanup-20260917/files/`, followed by
each original relative path. All 287 files are recoverable there; this local
archive is not included in the remote experiment snapshot. The prior 697-file
source backup is separate and remains unchanged.

## Retained intentionally

No simulator, planner, learner, UI, fitting, export, or test code was removed.
The manifest includes a 540-file Python source hash baseline for later checks.

Older correction utilities are not safely disposable merely because a generic
CLI exists. Current fitting jobs record source hashes, and the saved horizontal
reference explicitly hashes `tools/correct_m6_horizontal.py`. Removing such
files can break existing verification. No provenance records were rewritten to
hide those dependencies.

All recordings, model assets, commands, numerical results, original photographs,
manuscript sources, selected figure assets, Git history, and prior archives remain
in place. The live Dropbox manuscript was only checked for figure references;
it was not edited. Temporary experiment scripts and non-LaTeX logs were retained.

The current implementation was saved in commit `904c02e`, and the current flight
baseline in `819d3c96`. See `docs/data/CURRENT_FLIGHT_BASELINE_20260917.json` for
the retained models, command hashes, and validation limits. That preservation
pass initially retained shared SAC/PPO dependencies. The subsequent
[trainer retirement](LEGACY_TRAINING_RETIREMENT_20260917.md) separates the shared
numerical helpers and removes the trainers and policy launch paths.
