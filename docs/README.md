# AeroWhip documentation

## Write the paper and run the study

1. [Current handoff](../HANDOFF.md): project scope, evidence boundaries and jobs.
2. [Paper-writing handoff](PAPER_WRITING_HANDOFF.md): contribution, method and
   minimal figures/results to prepare.
3. [One-day experiment protocol](PAPER_EXPERIMENT_PROTOCOL.md): retained M0,
   new 20-flight chain, fixed roles and continuous-distance analysis.
4. [Architecture](ARCHITECTURE.md): current source and workflow map.

The `deployment` branch contains the operator app and its `docs/LAB_RUNBOOK.md`.
Use its setup and validation guides for the colleague's lab transfer. This
research checkout keeps the broader six-page interface and development evidence.

## Technical references

| Topic | Guide |
|---|---|
| Staged full-model fitting | [Frozen system identification](FROZEN_SYSTEM_IDENTIFICATION.md) |
| Command generation and execution | [Direct PVA workflow](DIRECT_PVA_WORKFLOW.md) |
| Loaded-UAV response | [Nominal drone pose response](NOMINAL_DRONE_POSE_RESPONSE.md) |
| Reference points and transforms | [Geometry and coordinates](GEOMETRY_COORDINATE_CONVENTIONS.md) |
| Data review and provenance | [Data lifecycle](DATA_LIFECYCLE_AND_RECORDING_GUIDE.md) |
| Environment | [Installation](INSTALL.md) |
| Research asset transfer | [Reproducibility](REPRODUCIBILITY.md) |
| Test scope | [Tests](../tests/README.md) |

These references retain experiment-specific implementation detail. Their dated
status and fresh-M0 proposals do not override the current experiment protocol.
Frozen copies inside completed jobs remain unchanged.

## Development evidence and literature

- [M0/M1/M2 system comparison](M0_M1_M2_SYSTEM_COMPARISON.md): existing 5/5/3
  development takes, original forecasts and common-flight diagnostics.
- [Full adaptation](FULL_MODEL_ADAPTATION.md) and
  [M2 regression analysis](M2_REGRESSION_ANALYSIS.md): what the model updates
  changed and why component improvement does not guarantee complete improvement.
- [Whole-system audit](PAPER_PIPELINE_AUDIT.md): implementation and literature
  review. Its former expanded experiment proposal is superseded.
- [Current literature review and paper framing](ICRA_2027_RELATED_WORK_AND_FRAMING.md),
  [detailed literature notes](RELATED_PAPERS_DETAILED_REVIEW.md) and
  [bibliography](related_papers.bib): check primary sources when writing claims.
- [M2 PPO exploration](M2_PPO_PERSISTENT_EXPLORATION.md): separate development
  work; consult actual run status before describing it.

The active documentation retains methods and recent evidence needed for the
paper. Superseded run diaries and unused old designs are removed from the main
workspace. Relevant development results explain design decisions; they are not
the new study's final outcomes.
