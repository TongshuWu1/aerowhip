# AeroWhip: current handoff

Updated 11 September 2026. Project/repository name: **AeroWhip / `aerowhip`**.
Paper title: **AeroWhip: Aerial Cable Whipping through Iterative Model Refinement**.

Read [the paper handoff](docs/paper/PAPER_WRITING_HANDOFF.md) and
[experiment protocol](docs/paper/PAPER_EXPERIMENT_PROTOCOL.md) for current decisions.
They supersede older proposals for new preliminary collection, a refitted M0,
multiple targets, 90 final flights or binary 5 cm paper outcomes.

## Current scope

- Retain the existing M0, preliminary measurements and their original roles.
  Collect a new whip chain: 5 M0 flights, 5 M1 flights, then 5 paired M0/M2 blocks
  (10 final flights), at one target. Do not refit M0.
- Use continuous minimum 3D tip-to-target distance over 0–1.5 s as the main task
  metric. Evaluate all three frozen models on the same final recordings for
  command-to-tip prediction RMS. Final recordings never enter fitting or tuning.
- Use the existing staged full update and frozen M0 planner settings. Retained
  M0 disables the cable residual; full M1/M2 enable it. Report this model-capacity
  difference; this is not a controlled extra-data-only comparison.
- The current PVA workflow exports 30 Hz CSVs. Aircraft execution uses the
  colleague's separate flight program. Preserve historical command semantics
  for historical artifacts; do not reinterpret older force/20 Hz policies.
- The main research UI has six pages. The `deployment` branch has the five-page
  operator workflow and its own setup/runbook. Ubuntu/RTX 5080 is the planned lab
  machine, not a validated environment. See that branch's `docs/VALIDATION.md`.

## Existing evidence

The historical flown lineage is **M0 → M1-full → M2-frozen-refit-v1**, with
5/5/3 development takes. The gain-only M1 is a sibling, not the full update;
the original full M2 and its frozen refit have the same model signature.
These observations informed system design and are not an untouched final test.
Read [the development comparison](docs/development/M0_M1_M2_SYSTEM_COMPARISON.md) and
[the frozen staged method](docs/methods/FROZEN_SYSTEM_IDENTIFICATION.md), retaining their
study-specific qualifications and original reporting windows.

M0's retained command is
`runs/rehearsals_pva/20260910-022818-648386-M0-development-whip`.
CSV SHA-256:
`ea2e20bba92eb55cad12559ea97fb817eed76cd9ef6b21c191bf7b504368dde3`.
Original forecast SHA-256:
`8ed2d231f333d567d71fd8871322e6a2d77b625a29ea2135c434553290b25e13`.
The research flight selection is recorded separately in
`config/pva/flight_selection.json`; do not change it as a documentation cleanup.

## Jobs and preserved work

PPO development is separate from the core MPPI paper study. Verified job
`runs/ppo_pva/20260911-142454-400090` completed at 2,097,152 attempts with stopping
reason `safety_ceiling`; that is not a convergence claim. Its best checkpoint is
from 2,080,768 attempts, SHA-256
`afe1826067889787c598aee00f4d0f6bb44f2fe77f67fb0e6daa257dc948e6ea`.
The completed recovery rehearsal is
`runs/rehearsals_pva/20260911-142454-400090-M2-ppo-persistent-whip`.
Preserve this run, checkpoints and recovery previews; do not duplicate or
restart it. Read actual status again before any later job action.
Two-target exploration is paused and the swing-and-settle task is retired.

The user authorized removing obsolete measurements, failed legacy trials, old
records and superseded documents, while keeping the recent work (especially
the 10 September M0/M1/M2 chain). Retain the complete current lineage, its
preliminary/calibration dependencies, current selected PPO/MPPI, new deployment
work, and the raw records, forecasts, weights, optimizer state and source
snapshots needed to explain and reproduce them. Age alone must not remove a
dependency of a retained model or result. Do not inspect, fit,
evaluate or plot the protected `fig8vertical_002` recording. Its opaque copying
for the previously authorized private lab transfer is allowed.

Cleanup does not change fitting roles, rewards, physics or the selected flight.
Do not reset the working tree, rewrite Git history, or restart old jobs while
organizing the project. Do not treat removed historical status prose as a live
instruction if it is encountered in a retained frozen snapshot.

The 11 September 2026 cleanup removed 5,669 obsolete or generated files
(78,039,454 bytes) across 153 reviewed paths. All 10,874 retained evidence files
matched their pre-cleanup hashes; the recent M0–M2 lineage, required preliminary/
calibration inputs and selected PPO/MPPI remain intact.

## Folder organization

Current recordings are in `data/flight_batches/`, handoff exports in `exports/`,
paper illustrations in `paper/figures/`, and the optional SDK in
`third_party/natnet/`. Documentation is grouped under `docs/setup`,
`docs/methods`, `docs/paper` and `docs/development`. See
[the folder map](docs/FOLDER_MAP.md).

Only paths and guides changed during organization; frozen experiment/config
bytes and model identities are preserved. Local compatibility links retain old
data/figure paths used inside historical evidence. New code uses the canonical
locations and does not depend on Windows links for new studies.
