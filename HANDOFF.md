# Deployment handoff

Updated 11 September 2026. Branch: deployment.

Start with README.md, then docs/LAB_RUNBOOK.md. The primary launcher is
run_lab.py; run_simulation.py opens the advanced research workspace.

This branch incorporates current first-party PVA source, including uncommitted
September 11 changes. It is isolated from the active research checkout. Its source
parent is recorded in SOURCE_SNAPSHOT.json. Historical data are not new evidence.

The user selected existing M0 and preliminary data as the baseline. Do not
recollect or refit M0. Collect five M0 flights, fit M1, collect five M1 flights,
fit M2, then collect five interleaved M0/M2 pairs. The primary endpoint is
continuous target distance; no 5 cm success cutoff is required.
Preserving M0 preserves its disabled cable residual, so M0â†’M2 is a system
refinement comparison including that capacity change.

The colleague uses Ubuntu/RTX 5080. Files and launchers must also work across
Windows/Linux checkouts. Preserve imported provenance and resolve runtime paths
locally. The private baseline bundle supplies retained M0 and preliminary replay
inputs. Recordings, jobs, exports and reports stay inside this repository.

The simulator only exports CSVs. Actual execution uses the colleague's existing
flight program. Sender integration, ROS topics and aircraft-control UI are outside
this branch. The lab operator verifies hardware/controller timing, measurement and
launch preparation.

See docs/VALIDATION.md for completed checks and remaining platform verification.
Installing requirements is not validation of the fitted dynamics or a maneuver.
