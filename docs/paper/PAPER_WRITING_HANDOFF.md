# AeroWhip paper-writing handoff

The active source is
`C:/Users/wts28/Lehigh University Dropbox/TonyLehigh Wu/Apps/Overleaf/AeroWhip-ICRA/main.tex`.
The repository `paper/manuscript.tex` is an older reference. Do not overwrite
Dropbox with it. Preserve the live section structure, author notes and LaTeX
notation. Cloud synchronization is not established by a local file edit.

The paper centers on sim-to-real-to-sim refinement of the quadrotor and cable
dynamics for aerial whipping. The implementation predicts loaded quadrotor
motion first, then drives the DDER cable with its rotated attachment trajectory.
Do not claim explicit two-way cable-force feedback to the quadrotor model.
The active M2 keeps both residuals, using the retained M1 drone residual.

Initial MPPI design uses the restored reference-shape objective and quintic
position B-splines. After each model update, local command correction uses the
same original M0 physical tip/quadrotor reference and fixed timestamps, starting
from the latest executed controls. Only initial MPPI uses correlated sampling;
local correction uses bounded regularized Gauss-Newton. Measured trajectories
influence commands through the fitted model, without a direct measured bias.

Keep fitting independent of the target/reference objective. Existing M1_003/005
recordings were used for M2 development selection, so do not say validation
never influences model choice or present these as independent final evidence.
The physical M0/M1 results and model-prediction results are distinct. Five M2-selected flights are now evaluated. See the [current handoff](../../HANDOFF.md).

The latest editorial changes preserve the framework heading structure and
17 numbered equations. Retain necessary variable definitions, model equations
and loss definitions; implementation settings belong in Experiments. Use
straightforward, concise research prose. The figure is a schematic, not sampled
physical evidence. The user will replace temporary experiment figures.

Current applied reviews: [framework editing](FRAMEWORK_EDITORIAL_REVISION_20260913.md),
[model-improvement framing](MODEL_IMPROVEMENT_FRAMING_20260913.md), and
[framework revision](OVERLEAF_FRAMEWORK_REVISION_20260913.md).

Earlier reviews, figure revisions and temporary manuscript backups were moved
to `delete/cleanup-20260913/`, preserving their repository-relative paths.
The complete previous handoff is in its `before_cleanup/docs/paper/` folder.
These historical files remain available for provenance; no manuscript was
compiled or substantively edited during cleanup.

The command-correction schematic is now integrated beside the local correction
text. It uses Figure 1's TikZ style and explicitly illustrative tip trajectories.
Automatic numbering is currently Figure 2, becoming Figure 3 after the planned
framework overview. See [the figure record](COMMAND_CORRECTION_FIGURE_20260913.md).

The command-correction figure now uses actual saved M2-selected forecasts and
the original reference, replacing the illustrative curves. Caption distinguishes
predictions from measurements; source hashes are recorded in
COMMAND_CORRECTION_FIGURE_DATA_20260913.json. No simulation was rerun.

The command-correction figure now uses the actual M0-to-M1 correction run
(M1-local-20260913-032256-816506), superseding the M2-selected example. Both panels
use M1 predictions, with M0/corrected M1 commands and the original M0 reference.
The red error connectors remain removed.

## Experiments section added

The live manuscript now includes Experiments, with setup, refinement/data roles,
metrics, full five-take physical comparison, a disclosed post-hoc subset of M2
takes 002/004, and development-only prediction results. All five M2 recordings
remain the primary analysis. No hardware defect is treated as established.
Abstract numbers use the full batches, not the selected pair. The source
bibliography and existing figures are unchanged. The prior manuscript is backed
up at `tmp/experiments_20260913/before/main.tex`.
