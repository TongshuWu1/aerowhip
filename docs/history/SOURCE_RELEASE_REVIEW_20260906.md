# Source review bundle verification — September 6, 2026

The source-only review archive is `../../dist/source-review-20260906.zip`, with
its SHA-256 checksum alongside it. The build contains 172 files and about 1.4 MB
of uncompressed source/configuration content. `RELEASE_MANIFEST.json` lists file
hashes, original and bundled configuration hashes, and portability changes.

Verification used the existing development Python environment while importing
source from the separate bundle directory:

- Short CPU hanging-hover simulation: finite state; maximum segment error
  approximately 1.39e-16 m.
- Point-mass, early-stop, flight-adaptation and source-release checks: 18 passed.
  Pytest reported a cache-directory permission warning; tests completed normally.
- Offscreen desktop startup: all five pages opened with an empty dataset;
  imports were checked to resolve inside the bundle directory.
- Release tests verify archive hashes, artifact exclusions and preservation of
  numerical physics, rewards, timing and the embedded action prior.

This is not a fresh-environment installation test, full-suite run, GPU training
benchmark, remote CI result or hardware validation. The GitHub CPU smoke workflow
is prepared but has not run remotely. Portable training defaults are smaller
than the workstation comparison; this transformation is explicitly recorded.

The development repository still contains legacy tracked artifacts and extensive
preexisting changes. No staging, commit, push or history rewrite was performed.
The source archive excludes measurements, checkpoints, paper PDFs, internal notes
and Git history. Existing SAC training and active configuration were preserved.

Before distribution, authors must decide ownership, authorship, software license,
third-party attribution and any separate data/checkpoint release. No authors,
license or DOI were invented. This is a review candidate, not a published artifact.
