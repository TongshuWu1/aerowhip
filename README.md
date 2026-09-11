# Aerial Whip Lab

A desktop workflow for planning a cable whip, exporting its PVA commands,
reviewing recorded flights, and refining the model between experiments.
The app exports files; your existing flight program executes them.

**Start the lab app with `run_lab.py`. No Codex session is required.**

## Quick start

Use Python 3.12. On the Ubuntu/NVIDIA lab computer:

```sh
python3 setup_lab.py --device cuda
sh start_lab.sh
```

On Windows:

```powershell
py -3.12 setup_lab.py --device cuda
.\start_lab.cmd
```

Setup creates `.venv/` inside this checkout. It installs the pinned desktop
dependencies and the official PyTorch CUDA 12.8 build. For a machine used only
to inspect data/UI, use `--device cpu`; the full fitting path requires CUDA.

The **private lab handoff ZIP** includes the retained M0 baseline and preliminary
inputs. Extract the entire ZIP, run setup, and open the app. A source-only checkout
instead starts with an empty baseline: use **Import baseline** and select the
retained-M0 bundle supplied with the experiment. Never copy an old virtual
environment between operating systems.

See [installation and troubleshooting](docs/INSTALL.md) if setup needs attention.

## Run the experiment

1. Create a named experiment using the retained M0.
2. Export the M0 CSV. Execute it with the external flight program and import each
   controller/OptiTrack recording pair into its assigned take.
3. Review the recordings, then run the M1 update and its MPPI plan.
4. Collect and review the five M1 takes, then run the M2 update and plan.
5. Complete the five interleaved M0/M2 pairs and export the comparison.

The app assigns three adaptation takes and two operational-validation takes in
each update batch. Final comparison recordings are never used for fitting.
The primary result is **minimum 3D tip-to-target distance**, with prediction RMS
reported separately. There is no binary 5 cm success requirement.

Keep the existing M0 and preliminary data. The new collection has **20 whip
executions at one target**. Follow the [one-day operator runbook](docs/LAB_RUNBOOK.md)
for the exact sequence and recording review.

## Where files go

| Folder | Contents |
|---|---|
| `workspace/baseline/` | Verified retained M0, original forecast, preliminary replay inputs |
| `experiments/<study>/` | Take schedule, recording pairs, review decisions and study ledger |
| `exports/<study>/<generation>/` | The exact CSV and accompanying frozen export files |
| `runs/` | Fitting, planning, rehearsal and evaluation outputs, with logs |
| `config/` | Portable source defaults and local catalogs |
| `deployment/` | Guided lab app, workflow and baseline packaging |
| `simulator/`, `planning/`, `experimental_data/`, `learning/` | Numerical and research implementations |
| `tests/` | Isolated verification; no hardware is required for the deployment subset |

Everything needed for a study stays inside the checkout. Copy the whole folder
when transferring a study, excluding `.venv/` and rebuilding it on the destination.
Do not move or edit a job while it is running. Generated/private experiment files
are excluded from source control; the private handoff archive includes the
baseline explicitly.

## Check the computer

```sh
.venv/bin/python tools/check_lab.py --compute --require-cuda --require-baseline
```

On Windows use `.venv\Scripts\python.exe` instead. This checks dependencies,
CUDA, a small float64 tensor operation and baseline integrity. It does not run a
fit, planner or flight. [Validation notes](docs/VALIDATION.md) distinguish completed
Windows checks from the remaining Ubuntu/RTX 5080 check.

## Research and reproduction

The guided app wraps the existing staged identification and offline MPPI code.
`python run_simulation.py` opens the advanced research interface, including PPO;
it is not required for the colleague's experiment. The historical
`--headless` option runs a different force-model diagnostic and is not the lab
startup test.

Read [method and evidence notes](docs/REPRODUCIBILITY.md). Preserving the original
M0 retains its disabled cable residual; the later full updates add that capacity.
The M0-to-M2 experiment therefore evaluates the complete refinement procedure.

This branch provides software and CSV exports. It does not implement the external
aircraft sender or establish vehicle-specific execution limits. The operator uses
the laboratory's existing execution, measurement and launch procedures.
