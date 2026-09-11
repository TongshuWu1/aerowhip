# Source and lab packages

The main research source exporter is `tools/build_source_release.py`. It copies
the current application and linked documentation, with empty experiment
catalogs and no raw recordings or fitted checkpoints. It does not reproduce
historical results without separately supplied research assets.

The `deployment` branch contains the colleague-facing operator app and
`tools/build_lab_release.py`. Its private package can include the retained
M0/preliminary baseline; its source package requires a baseline import.

Use the [installation guide](../setup/INSTALL.md),
[reproducibility guide](../setup/REPRODUCIBILITY.md), and
[paper handoff](../paper/PAPER_WRITING_HANDOFF.md) for the current workflow.
Packaging checks do not establish Ubuntu or aircraft validation.
