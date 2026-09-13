# Deployment handoff

## 12 September: new travelling-fold MPPI task

The user explicitly replaced the MPPI objective for the upcoming study. New
studies use `config/pva/systematic_strike.json`: accurate, fast forward tip motion
with a required travelling fold, plus the existing flight/recovery constraints.
Keep the calibrated M0 model and preliminary data, but generate a new M0 command.
Freeze the new objective, fold criterion, templates and budget across M0/M1/M2.
Existing studies and historical commands keep their original definitions.
See [the task definition](docs/methods/TARGETED_FOLD_STRIKE.md) and its validation
record before using the new workflow. This update supersedes instructions to
reuse the historical M0 command for a newly created study.


Updated 11 September 2026. Branch: deployment.

Start with README.md, then docs/lab/LAB_RUNBOOK.md. The primary launcher is
run_lab.py; run_simulation.py opens the advanced research workspace.

This branch shares the committed first-party implementation with main while
keeping portable lab configurations and private experiment files separate.
SOURCE_INTEGRATION.json records consolidation bases; SOURCE_SNAPSHOT.json is
historical provenance. See docs/GIT_WORKFLOW.md. Historical data are not new evidence.

The user selected existing M0 and preliminary data as the baseline. Do not
recollect or refit M0. Collect five M0 flights, fit M1, collect five M1 flights,
fit M2, then collect five interleaved M0/M2 pairs. The primary endpoint is
continuous target distance; no 5 cm success cutoff is required.
Preserving M0 preserves its disabled cable residual, so M0 to M2 is a system
refinement comparison including that capacity change.

The colleague uses Ubuntu/RTX 5080. Files and launchers must also work across
Windows/Linux checkouts. Preserve imported provenance and resolve runtime paths
locally. The private baseline bundle supplies retained M0 and preliminary replay
inputs. Recordings, jobs, exports and reports stay inside this repository.

The simulator only exports CSVs. Actual execution uses the colleague's existing
flight program. Sender integration, ROS topics and aircraft-control UI are outside
this branch. The lab operator verifies hardware/controller timing, measurement and
launch preparation.

See docs/lab/VALIDATION.md for completed checks and remaining platform verification.
Installing requirements is not validation of the fitted dynamics or a maneuver.
