# Command-correction schematic added

## Superseding update: actual saved predictions

The user requested actual data instead of illustrative curves. The live figure
now plots all 171 saved tip samples over 0--34/30 s from correction job
`M2-selected-local-20260913-042637-826368`. The before panel uses that job's
`baseline_prediction.npz` (executed M1 command under M2-selected); the after
panel uses its original saved `rehearsal.npz` forecast. Both show the byte-verified
original M0 reference and the same fixed M2-selected model. All paths use the
same world x-z axes and equal spatial scale, without smoothing or retiming.
Four markers sample quarter-interval times using the closest stored samples.
Caption and text identify saved simulation predictions, not physical measurements.

3D tip-reference RMSE recomputed from the saved arrays is 0.05502560897455247 m
before and 0.03960350448060721 m after, matching the original job report. These
numbers verify the arrays; they are not new physical outcomes. No simulation,
fitting or optimization was run. Original evidence files remain unchanged.
Source hashes and data checks are in `COMMAND_CORRECTION_FIGURE_DATA_20260913.json`.
Build, source backups and compile checks are in `tmp/correction_data_figure_20260913`.
The previous schematic described below is superseded. Automatic numbering is
still Figure 2 until the framework overview is inserted.

## Historical schematic

Created `figures/command_correction.tex` in the live Dropbox manuscript using
the existing Figure 1 TikZ palette and vector workflow. The two panels show
latest and corrected commands evaluated under the same fixed refined model.
Dashed gray reference tip paths and their timestamps are identical; teal curves
are illustrative predictions and red segments join matching-time samples.
The curves are tip paths, not cable centerlines. They contain no measured or
simulated result and illustrate the intended effect of the tracking objective.

Inserted a single-column figure near command correction and integrated its
cross-reference into the opening explanation. Caption explicitly says schematic.
The current manuscript contains no framework-overview figure, so automatic
numbering is Figure 2. It becomes the proposed Figure 3 when that overview is
inserted earlier. No counters were manually skipped and no placeholder inserted.

Compiled on Windows with Tectonic 0.17.0 and rendered with Poppler. Inspected
the standalone artwork and all seven manuscript pages. Equations (17), heading
structure, existing figures, bibliography and author notes are unchanged.
No overflow or unresolved references; two mild existing underfull paragraphs.
No numerical tests, fitting, planning, or physical evaluations were run.

Live files: Dropbox `main.tex` and `figures/command_correction.tex`.
Preview: `output/pdf/AeroWhip_framework_revision_20260913.pdf`.
Artwork preview: `output/pdf/AeroWhip_command_correction.png`.
Backup and checks: `tmp/correction_figure_20260913/publication.json`.
Source concurrency checks passed. Cloud synchronization remains unverified.

Latest style change: removed red same-time error connectors and their legend
at the user's request. Reference/prediction paths, markers, axes, and data are
unchanged. Verified figure and page 5 after rebuilding; no overfull boxes or
unresolved references. Backup: tmp/correction_legend_20260913.

Latest data selection: the user requested M0 and M1. The figure now uses actual
saved predictions from M1-local-20260913-032256-816506 and its original rehearsal.
Both panels use the fixed M1 model: M0 commands before, corrected M1 commands
after. The dashed reference is the original M0 forecast. No M2 data was relabeled.
Red connectors remain absent. 171 samples and matched quarter-interval markers;
3D reference RMSE matches original report (0.1441143925289969 m before,
0.03734728569844124 m after). These are prediction errors. Caption and text updated;
source provenance refreshed. No rerun or measured data used. Checked compilation
and affected pages. Backup: tmp/correction_m0_m1_20260913.
