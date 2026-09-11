# AeroWhip documentation

See the [folder map](FOLDER_MAP.md) for data, exports and run locations.

- `setup/`: installation, lab setup and transfer.
- `methods/`: model, fitting, geometry and evaluation definitions.
- `paper/`: writing, protocol, literature and publication readiness.
- `development/`: retained exploratory results and diagnostics.
- `history/`: necessary older technical references.

## Write the paper and run the study

1. [Current handoff](../HANDOFF.md): project scope, evidence boundaries and jobs.
2. [Paper-writing handoff](paper/PAPER_WRITING_HANDOFF.md): contribution, method and
   minimal figures/results to prepare.
3. [One-day experiment protocol](paper/PAPER_EXPERIMENT_PROTOCOL.md): retained M0,
   new 20-flight chain, fixed roles and continuous-distance analysis.
4. [Architecture](ARCHITECTURE.md): current source and workflow map.

The `deployment` branch contains the operator app and its `docs/lab/LAB_RUNBOOK.md`.
Use its setup and validation guides for the colleague's lab transfer. The
advanced six-page interface is also available. Development guides describe the
separately held research evidence in main; they do not select a lab model.

## Lab operator guides

Start the five-page app with `python run_lab.py`. Follow the
[operator runbook](lab/LAB_RUNBOOK.md), [lab setup](lab/INSTALL.md), and
[validation record](lab/VALIDATION.md). The `deployment` branch retains the
curated lab defaults; `main` also includes these tools alongside research.

## Technical references

| Topic | Guide |
|---|---|
| Staged full-model fitting | [Frozen system identification](methods/FROZEN_SYSTEM_IDENTIFICATION.md) |
| Command generation and execution | [Direct PVA workflow](methods/DIRECT_PVA_WORKFLOW.md) |
| Loaded-UAV response | [Nominal drone pose response](methods/NOMINAL_DRONE_POSE_RESPONSE.md) |
| Reference points and transforms | [Geometry and coordinates](methods/GEOMETRY_COORDINATE_CONVENTIONS.md) |
| Data review and provenance | [Data lifecycle](methods/DATA_LIFECYCLE_AND_RECORDING_GUIDE.md) |
| Environment | [Installation](setup/INSTALL.md) |
| Research asset transfer | [Reproducibility](setup/REPRODUCIBILITY.md) |
| Test scope | [Tests](../tests/README.md) |

These references retain experiment-specific implementation detail. Their dated
status and fresh-M0 proposals do not override the current experiment protocol.
Frozen copies inside completed jobs remain unchanged.

## Development evidence and literature

- [M0/M1/M2 system comparison](development/M0_M1_M2_SYSTEM_COMPARISON.md): existing 5/5/3
  development takes, original forecasts and common-flight diagnostics.
- [Full adaptation](methods/FULL_MODEL_ADAPTATION.md) and
  [M2 regression analysis](development/M2_REGRESSION_ANALYSIS.md): what the model updates
  changed and why component improvement does not guarantee complete improvement.
- [Whole-system audit](paper/PAPER_PIPELINE_AUDIT.md): implementation and literature
  review. Its former expanded experiment proposal is superseded.
- [Current literature review and paper framing](paper/ICRA_2027_RELATED_WORK_AND_FRAMING.md),
  [detailed literature notes](paper/RELATED_PAPERS_DETAILED_REVIEW.md) and
  [bibliography](paper/related_papers.bib): check primary sources when writing claims.
- [M2 PPO exploration](development/M2_PPO_PERSISTENT_EXPLORATION.md): separate development
  work; consult actual run status before describing it.

The active documentation retains methods and recent evidence needed for the
paper. Superseded run diaries and unused old designs are removed from the main
workspace. Relevant development results explain design decisions; they are not
the new study's final outcomes.
