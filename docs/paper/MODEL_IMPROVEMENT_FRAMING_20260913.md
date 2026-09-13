# Applied model-refinement framing

The user clarified that AeroWhip refines the vehicle-and-cable dynamics model,
not only the command for one task. The live Dropbox manuscript now emphasizes
model refinement for command-to-tip prediction and subsequent planning.
Fixed-reference local command correction is an application of that updated
model. Prediction accuracy and physical correction outcomes remain separate
evaluations. No generalization or numerical improvement is asserted.

The fitting description explicitly distinguishes measured-motion losses from
the target and planned reference. Physical constants that remain fixed are
still identified. The current M2 CLI confirms initialization from the latest
executed M1 controls; the correction prose now describes latest-executed-command
initialization, while the objective retains the original M0 reference.

Suresh and Atkeson now receive a specific related-work sentence explaining
measured task errors and local robot/rope inversion. The existing citation key
`suresh2026learning` and published RSS bibliography entry were retained, verified
against https://www.roboticsproceedings.org/rss22/p126.html. The supplied earlier
arXiv entry was not duplicated. Their model-based action update is distinguished
from our dynamics-model refinement without claiming an absence of a model in ILC.

Validation: Tectonic compilation and seven-page visual inspection on Windows;
17 equation bodies, headings, author notes, bibliography and citation-key set
preserved. No overfull boxes or unresolved references; two mild existing
underfull paragraphs. Implementation, experiments and numerical settings unchanged.

Backup, staged sources and validation manifest:
`tmp/model_improvement_framing_20260913/`.
Updated preview: `output/pdf/AeroWhip_framework_revision_20260913.pdf`.
The live Dropbox `main.tex` was updated with a concurrent-edit check. Cloud
synchronization was not verified.
