# Complete application port for simulator experiments

Latest user correction: retain the entire UI and all training/policy/model code.
The earlier reduced single-scene UI was discarded and the full application was
restored. The seven original pages and their subpanels remain available.

Source: twin-rewrite commit 19dc9a2694f4edafc0ebf94482139ca16e334b2c.
That original checkout and research artifacts are untouched. Simulator has fresh
root history and a separate checkout.

No original recordings, processed data, learned policies/checkpoints, fitted
models/residual weights, or historical run outputs were copied. Startup uses
an explicitly unfitted nominal profile; both NNs are disabled. Small changes
allow nominal-only asset loading, prediction, and snapshots without fake weights.
The model page identifies this profile as unfitted, not an already collected M0.

Next requested direction: independent Isaac Lab/PhysX plant, preliminary data
collection, then new M0 identification. No PhysX plant was built, no preliminary
data collected, and no fitting or training started during the branch port.
The existing external-model Isaac host and five-take adaptation runner remain
code foundations, not completed independent-plant/preliminary-fit workflows.

Verification: 37 focused physics/geometry/FullState/nominal-only checks passed.
All seven UI pages and their sub-tabs opened with Qt offscreen; empty libraries
and disabled continuation/fit-without-data controls verified. Original checkout
remains clean. This is not an Isaac/PhysX benchmark or a full historical suite.
