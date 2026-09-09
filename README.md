# Sim → Real → Sim aerial whipping

The current workflow generates **desired position, velocity and acceleration at 30 Hz**, using either PPO or independent MPPI. Bounded XYZ jerk integrates into a consistent P/V/A reference. The fitted loaded-drone model and drone NN predict tracked-origin motion; its rotated cable attachment drives DDER and the cable NN. There is no virtual-force planning stage in new PVA jobs.

Generation is offline and open loop. Start from a settled hover with an assumed hanging cable, freeze the complete command sequence, then export whip + smooth recovery + final hold. The real controller continues to use cmdFullState. Saved predictions are matched to actual flights using their exact CSV, not regenerated with a later model.

Read [the current PVA design](docs/DIRECT_PVA_WORKFLOW.md) and [current status](HANDOFF.md). Historical force policies and old 20/30 Hz workflows remain preserved; their checkpoints cannot be reinterpreted as jerk policies. Independent Isaac/PhysX development is paused.

For manuscript work or onboarding a separate writing agent, start with the detailed [paper-writing handoff](docs/PAPER_WRITING_HANDOFF.md): technical method, equations, candidate contributions, experiment lineage, verified results, limitations, and source/artifact links.

## Start here

For the current Windows/Ubuntu lab transfer, read [HANDOFF.md](HANDOFF.md) and
[lab setup](docs/LAB_SETUP.md). Fitting and PPO remain stopped; the latest MPPI
result is completed historical-model simulation. Deployment packages retain their
saved action/model semantics; the real flight interface still needs verification.

For a new environment, start with [installation](docs/INSTALL.md). For a source-only
research release, see [publication preparation](docs/PUBLICATION.md). The older
source-only release builder has not been verified for the complete current PVA workflow.

```powershell
.venv/Scripts/python.exe run_simulation.py
```

The UI has six pages: **Models & fitting**, **Recordings**, **PPO**, **MPPI**, **Rehearsals**, and **Flight comparison**. PPO and MPPI own separate setup files, run libraries and rehearsal/export tabs. Model-fit diagnostics are distinct from prospective measured-flight results. Older documents below describe preserved historical stages; the current PVA design and HANDOFF take precedence.

- [Flight/adaptation workflow](docs/FLIGHT_ADAPTATION_QUICKSTART.md)
- [Current decisions and completed simulation](HANDOFF.md)
- [Research proposal](docs/RESEARCH_PROPOSAL_ADAPTIVE_AERIAL_WHIP.md)
- [Documentation index](docs/README.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Reproducibility and data availability](docs/REPRODUCIBILITY.md)
- [Command-line tools](tools/README.md)

## Repository layout

| Folder | Purpose |
|---|---|
| `config/` | Active model, task and algorithm defaults |
| `simulator/` | DDER physics, modeled drone response, replay and desktop UI |
| `learning/` | PPO, rewards and execution; preserved legacy learning code |
| `planning/` | Independent MPPI optimization and saved-plan generation |
| `deployment/` | Rehearsal, recovery, CSV export and portable replay |
| `experimental_data/` | Recording processing, calibration and flight adaptation |
| `tools/` | Current workflow commands, evaluation and research utilities |
| `tests/` | Checks grouped into physics, calibration, training, flight and UI |
| `docs/` | Current guides and research design; dated reports in `docs/history/` |
| `data/` | Raw recordings, processed data, physical baselines and flight imports |
| `runs/` | Training checkpoints and per-run validation records |
| `results/` | Comparison studies, immutable source snapshots and plots |

Ignore rules exclude ordinary generated output and caches. Selected research
artifacts are explicitly tracked, including Git LFS assets; use the reproducibility
guide when transferring the project. Raw recordings and historical outputs are preserved.

## Run checks

```powershell
.venv/Scripts/python.exe run_tests.py physics
.venv/Scripts/python.exe run_tests.py flight
.venv/Scripts/python.exe run_tests.py training ui
.venv/Scripts/python.exe run_tests.py
```

With no group, all maintained tests run. See [the test guide](tests/README.md) for scope. Some calibration, GPU and training integration checks are intentionally slower.

## Reproducibility

Saved runs retain their model/task/algorithm settings, checkpoints and validation
history. Exported experiments retain the exact command CSV and prediction used
before flight. Changing defaults affects future jobs rather than rewriting past
experiments. See [reproducibility](docs/REPRODUCIBILITY.md) for evidence boundaries
and [historical documentation](docs/history/README.md) for superseded reports.
