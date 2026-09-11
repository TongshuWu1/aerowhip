# AeroWhip

**AeroWhip: Aerial Cable Whipping through Iterative Model Refinement**

Repository: [TongshuWu1/aerowhip](https://github.com/TongshuWu1/aerowhip).

AeroWhip combines a loaded-UAV response model, distributed cable dynamics and
offline trajectory planning. Recorded flights refine the model between trials;
each planned maneuver is exported as a frozen 30 Hz position, velocity and
acceleration (PVA) CSV for the laboratory's separate flight program.

The paper studies the complete plan–fly–measure–refine loop. Its main outcome is
continuous minimum 3D tip-to-target distance, supported by prediction error on
common recordings. The existing M0 and preliminary recordings are retained.
Historical M0/M1/M2 trials are development evidence; the planned new study uses
20 executions at one target.

## Start here

- [Current decisions and evidence boundaries](HANDOFF.md)
- [Paper-writing handoff](docs/PAPER_WRITING_HANDOFF.md)
- [One-day experiment protocol](docs/PAPER_EXPERIMENT_PROTOCOL.md)
- [Architecture and source map](docs/ARCHITECTURE.md)
- [Documentation index](docs/README.md)

## Research and lab versions

The main research checkout keeps six pages: **Models & fitting**, **Recordings**,
**PPO**, **MPPI**, **Rehearsals**, and **Flight comparison**. Run it from the
repository root using the existing environment:

```sh
# Linux
.venv/bin/python run_simulation.py
```

```powershell
# Windows
.venv/Scripts/python.exe run_simulation.py
```

See [installation](docs/INSTALL.md) for environment setup. Opening the interface
does not start training, fitting or planning.

The separate **`deployment` branch** provides the five-page operator app,
`setup_lab.py`, `run_lab.py`, and its `docs/LAB_RUNBOOK.md`. It keeps studies and
generated CSVs in the checkout; exports go to `exports/<study>/<generation>/`.
The private colleague bundle contains the retained M0/preliminary assets. These
assets are not included in the source-only branch, so a source-only user must
import the baseline separately. Neither version sends commands to the aircraft.

## Repository layout

| Path | Purpose |
|---|---|
| `config/` | Active configuration and experiment catalogs |
| `simulator/` | Loaded-aircraft response, geometry, cable physics and research UI |
| `planning/` | Offline MPPI search, PPO trajectory generation and immutable jobs |
| `learning/` | Shared PVA environment, objectives and learning implementations |
| `experimental_data/` | Recording review, staged fitting and model comparison |
| `deployment/` | Saved rehearsal, recovery and CSV export |
| `tools/` | Workflow and analysis entry points |
| `tests/` | Physics, fitting, flight, training and UI checks |
| `docs/` | Active paper and implementation guides |
| `data/`, `rehearsal_csv_and_result_in_real_flight/` | Measurements and imported recordings |
| `runs/` | Fitted models, saved commands, original forecasts and audit evidence |
| `output/` | Exported paper figures/PDFs and their provenance |

Retained evidence focuses on the current work and the recent M0/M1/M2 lineage,
including the preliminary inputs and calibration required to reproduce it.
Obsolete measurements, failed legacy runs and superseded guides are outside the
active paper workspace. Saved evidence retains its original model, settings,
source and data identities.
Do not regenerate an old forecast with a newer model and call it the original.
Use the [reproducibility guide](docs/REPRODUCIBILITY.md) when transferring research
assets; ordinary generated outputs are not necessarily tracked by Git.

For checks, see [the test guide](tests/README.md) and run the subset relevant to a
change. Report the actual environment tested. Existing Windows/RTX 4080 checks
do not establish Ubuntu/RTX 5080 validation.
