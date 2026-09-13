# Framework editorial revision

Applied the user's attached framework (`2f40f402-941b-4ab8-9921-5304fe503ab1`),
which matches the live source apart from whitespace, as an editorial review of
Section III. The structure, equation bodies, labels, author notes, bibliography,
and material outside the framework are preserved. All 17 numbered equations in
the manuscript and the unnumbered mathematical displays remain unchanged.

Changes:
- Explain the sequence once in the overview: initial plan, measurements, model
  refinement, local correction; retain the model for later prediction/planning.
- Shorten vehicle, attachment, residual-network, and initialization prose without
  claiming instantaneous cable reaction feedback to the vehicle.
- Distinguish the historical shape prior (`ref`) from the original MPPI reference
  (`0`); define the correction horizon by the original plan, not the latest seed.
- State that corrections use refined-model predictions, start from the latest
  executed command, and damp bounded increments in scaled control coordinates.
- Shorten motion/recovery descriptions and repeated evaluation explanations.
- Retain model fitting to measured motion independently of target/reference.
- Correct the outdated statement that validation never affects model selection.
  Current HANDOFF.md and docs/data/M2_SELECTED_20260913.md establish that M1_003/005
  were used for saved-model selection and are now development data. Main text
  describes development selection and reserves final test recordings. The exact
  saved-component selection and data roles must be documented in Experiments.
  No study roles, selected models, or fitting code were changed here.

The section shrank from 2,950 to 2,728 whitespace tokens including LaTeX (~7.5%).
Windows Tectonic compilation passed: seven pages, no overfull boxes or unresolved
references. Two mild pre-existing underfull paragraphs remain. All pages were
rendered with Poppler and inspected; affected pages were rechecked after final
wording edits. No numerical tests were run for this prose-only change.

Applied live file:
`C:/Users/wts28/Lehigh University Dropbox/TonyLehigh Wu/Apps/Overleaf/AeroWhip-ICRA/main.tex`.
Complete section export: `output/paper/AeroWhip_Framework.tex`.
Updated preview: `output/pdf/AeroWhip_framework_revision_20260913.pdf`.
Backup, staging, and verification: `tmp/framework_edit_20260913/`.
Concurrent source/PDF checks passed before applying. Cloud sync unverified.
