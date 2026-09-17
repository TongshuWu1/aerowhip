# AeroWhip documentation

See the [folder map](FOLDER_MAP.md) for repository organization.

- [Model catalog](../config/evaluation/campaign.json): registered models and flight metadata.
- [Baseline preservation](development/REPOSITORY_BASELINE_20260917.md): minimal cleanup scope and recovery locations.
- [Historical M0-reset protocol](paper/PAPER_EXPERIMENT_PROTOCOL.md): retained collection instructions, not the current model selection.
- [Fixed-reference correction](methods/FIXED_REFERENCE_CORRECTION.md) and [local correction](methods/M1_LOCAL_COMMAND_CORRECTION.md).
- [Model fitting](methods/FROZEN_SYSTEM_IDENTIFICATION.md) and [coordinate conventions](methods/GEOMETRY_COORDINATE_CONVENTIONS.md).
- [Position B-spline planner](methods/POSITION_SPLINE_PLANNER.md).
- [Lab runbook](lab/LAB_RUNBOOK.md) and [installation](setup/INSTALL.md).
- [Related work](paper/ICRA_2027_RELATED_WORK_AND_FRAMING.md) and [bibliography](paper/related_papers.bib).
- [Historical M2 selection](data/M2_SELECTED_20260913.md).

Superseded documents and temporary manuscript revisions are in
`delete/cleanup-20260913/` at the repository root. Their original paths are
preserved there. Older numerical and workflow guides retain their historical context.
In particular, generation-specific guides and older architecture descriptions
should not override the current catalog, saved job configuration, or implementation.
