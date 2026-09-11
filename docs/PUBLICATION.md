# Publication preparation

The source archive is a **review candidate**, not an automatically licensed or published release. No license, author identity, DOI, conference acceptance or artifact result is invented by the build process.

Before public distribution, the authors need to confirm project ownership, authorship, the intended software license, and third-party attribution. No reproduction of externally sourced paper PDFs, fonts or SDK archives is included. Dependencies are installed separately. Selecting a software license and clearing data redistribution remain owner decisions.

## Two different release scopes

1. **Source-only release:** `tools/build_source_release.py` includes the current PVA planner, fitting, comparison and UI packages with unfitted structural templates and an empty model/flight catalog. It has a file-hash manifest and archive checksum. Tests open the exported UI outside the live checkout. It does not include the fitted assets needed to reproduce a selected trajectory.
2. **Experiment artifact:** separately select the frozen source/configuration, models, approved datasets and raw metrics needed for a specific claim. Include data/checkpoint availability, acquisition/search budgets, validation selection and environment details. This has not been assembled or approved for public distribution yet.

The development repository was committed and pushed with selected research
artifacts and Git LFS assets on 9 September 2026. This development snapshot and
a curated paper release have different scopes. Ignore rules do not remove tracked
files or their history; select and document the evidence needed for each paper claim.

## Source-review builder

```powershell
python tools/build_source_release.py --output dist/source-review-new
```

Use a fresh output name. Inspect `PUBLICATION_METADATA.json`, `RELEASE_MANIFEST.json` and the `.zip.sha256` file. The builder checks selected personal-path/token/private-key patterns and unexpected large files; it is not a complete secret or licensing audit. Source hashes and numerical configuration invariants are checked in the release tests.

After checking author/license/data decisions, validate installation in a new environment and run supported-platform tests. Local verification using an existing interpreter is reported as such; it is not evidence that dependency installation on an unrelated machine succeeded.
