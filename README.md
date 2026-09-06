# Sim → Real → Sim aerial whipping

A force-controlled drone point coupled to a DDER cable, with PPO/SAC maneuver planning and between-flight model adaptation.

The current strike is **open-loop**: estimate the initial drone and cable state, generate one force sequence, execute it once, then return to PID hover at the frozen cutoff. Cable/hit feedback does not alter the sequence during execution. Commands are **20 Hz**, physics is **100 Hz**, and the maximum strike horizon is **1 second**. Exact current physics, rewards and limits live in `config/`; existing runs retain their own snapshots.

The real Lee-controller interface and force response are not yet validated. Adaptation exports are simulation candidates, not hardware-authorized commands.

## Start here

For the current Windows/Ubuntu lab transfer, read [HANDOFF.md](HANDOFF.md) and
[lab setup](docs/LAB_SETUP.md). The selected PPO can be transferred without retraining.
The [small deployment package](deployment/README.md) plans offline; the real ROS
controller bridge remains to be implemented and verified with the colleague.

For a new environment, start with [installation](docs/INSTALL.md). For a source-only
research release, see [publication preparation](docs/PUBLICATION.md). The release
builder creates a portable review bundle without copying recordings, checkpoints,
internal history or the current Git repository.

```powershell
.venv/Scripts/python.exe run_simulation.py
```

The UI has five pages: **Data & Calibration**, **Task & Rewards**, **PPO**, **SAC**, and **Real-world Updates**. PPO/SAC each have their own validation plots and 3D replay. The adaptation page imports flight data, compares replay, fits a physical candidate, and refines a force sequence without changing actor weights.

- [Tomorrow’s flight/adaptation guide](docs/FLIGHT_ADAPTATION_QUICKSTART.md)
- [Current decisions and active comparison](docs/PROJECT_CONTEXT.md)
- [Research proposal](docs/RESEARCH_PROPOSAL_ADAPTIVE_AERIAL_WHIP.md)
- [Documentation index](docs/README.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Reproducibility and data availability](docs/REPRODUCIBILITY.md)
- [Command-line tools](tools/README.md)

## Repository layout

| Folder | Purpose |
|---|---|
| `config/` | Active model, task and algorithm defaults |
| `simulator/` | DDER physics, point-force dynamics, replay and desktop UI |
| `learning/` | PPO/SAC, rewards, open-loop execution and sequence correction |
| `experimental_data/` | Recording processing, calibration and flight adaptation |
| `tools/` | Current workflow commands, evaluation and research utilities |
| `tests/` | Checks grouped into physics, calibration, training, flight and UI |
| `docs/` | Current guides and research design; dated reports in `docs/history/` |
| `data/` | Raw recordings, processed data, physical baselines and flight imports |
| `runs/` | Training checkpoints and per-run validation records |
| `results/` | Comparison studies, immutable source snapshots and plots |

Generated runs, results and caches are excluded from ordinary Git/search discovery. Raw recordings, active fit dependencies, selected policies, current training runs and the original CEM prior are preserved.

## Run checks

```powershell
.venv/Scripts/python.exe run_tests.py physics
.venv/Scripts/python.exe run_tests.py flight
.venv/Scripts/python.exe run_tests.py training ui
.venv/Scripts/python.exe run_tests.py
```

With no group, all maintained tests run. See [the test guide](tests/README.md) for scope. Some calibration, GPU and training integration checks are intentionally slower.

## Reproducibility

Each training run keeps its model/task/algorithm settings, checkpoints and validation history. The active comparison also freezes its source. Applying a new baseline or changing UI settings affects future runs rather than rewriting existing experiments. Historical reports document the settings at the time; the current configuration and project context take precedence.

Legacy UI pages, MPCC code/configuration, the duplicate launcher, one-off experiment scripts and obsolete tests have been removed from the working tree. Retired artifacts and the older source-review bundle are outside this repository, in the sibling `Sim2Real2SimWhip-retired-20260906` directory; `moved.json` there records their original locations. The current interface uses only the five research pages described above.
