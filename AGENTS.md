# Deployment branch maintenance

Read HANDOFF.md and README.md first. This branch provides a desktop workflow for
the preserved M0  to  new M1  to  new M2 experiment. The research checkout and original
evidence are separate; never modify or delete them as deployment cleanup.

- Preserve retained M0, preliminary train/holdout roles, original measurements,
  selected policies and frozen CSV/forecast bytes. Never access protected
  fig8vertical_002.
- New whip batches use 001/002/004 for adaptation and 003/005 for operational
  validation. Final M0/M2 pairs are never training data.
- Primary reporting is continuous minimum 3D tip-to-target distance. Do not change
  physics, rewards or fitting definitions as a UI or portability fix.
- All runtime files belong inside this checkout. Use pathlib and the current
  interpreter, not workstation paths or Windows-only interpreter assumptions.
- Exports are 30 Hz desired PVA CSVs. This application does not communicate with,
  arm or control an aircraft. Keep the actual flight program separate.
- Fit, search and training require explicit operator actions. Startup, refresh,
  health checks and import must not launch those jobs.
- Preserve immutable inputs, failed runs, model ancestry and preflight forecasts.
  Test changes with isolated fixtures. Record the actual OS/GPU tested; Windows
  checks do not establish Ubuntu/RTX 5080 validation.
- Keep the lab interface concise. Advanced research tools can remain available
  without becoming required steps for the colleague.

Maintain `main` and `deployment` as the two active branches. Keep shared source
synchronized and retain portable lab defaults during merges. See docs/GIT_WORKFLOW.md.
