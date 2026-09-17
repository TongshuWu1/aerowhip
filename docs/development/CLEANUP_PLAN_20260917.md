# Deeper cleanup review — September 17, 2026

Status: **planned removals were not executed**. The execution environment blocked
the combined move/delete operation before it started. The existing files remain
at their original locations. Only scoped LaTeX build-metadata ignore rules were
added to `.gitignore`, alongside this note and the local manifest.

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

The intended recovery destination is `archive/deep-cleanup-20260917/files/`,
followed by each original relative path. **The manifest is not a backup of these
287 files: the planned moves did not run.** The prior 697-file source backup is
separate and remains unchanged.

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

No commit, push, or merge was performed during this cleanup review.
