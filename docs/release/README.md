# Sim → Real → Sim aerial whipping

Research software for a force-controlled drone point coupled to a DDER cable, PPO/SAC maneuver planning, and between-trial physical-model adaptation.

**Status:** simulation research and offline adaptation prototype. A real Crazyflie/Lee-controller bridge and real-flight adaptation are not validated. This source-only review bundle contains no experimental recordings or trained checkpoints. Authorship and software-license metadata must be completed before public release.

## Quick start

Use Python 3.12 on Windows, create a virtual environment, and install the dependencies:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe run_simulation.py --headless --device cpu --duration 0.05
.venv/Scripts/python.exe run_simulation.py
```

See [installation](docs/INSTALL.md) for GPU selection, supported scope and smoke tests. The UI contains Data & Calibration, Task & Rewards, PPO, SAC and Real-world Updates. Its Execute button controls simulation, not hardware.

## Method

Estimate the initial drone/cable state, query the policy through a private simulator rollout, freeze one world-force sequence, execute once, then return to hover control at the planned cutoff. Commands are 20 Hz; physics is 100 Hz; the maximum strike horizon is one second. Actual hit/cable feedback does not change the strike in progress.

Offline adaptation compares measured-attachment and force-driven replay, proposes bounded cable-drag updates, and locally refines forces without retraining the actor. Candidate models and commands are saved for review and are not automatically applied to a vehicle.

## Layout and documentation

- `simulator/`: DDER, point-force dynamics, replay and desktop UI.
- `learning/`: PPO, SAC, shared task/rewards and sequence correction.
- `experimental_data/`: processing, fitting, flight recording and import.
- `config/`: portable defaults with embedded action prior.
- `tests/`: physics, calibration, training, flight and UI checks.
- [Architecture](docs/ARCHITECTURE.md)
- [Data and reproducibility](docs/REPRODUCIBILITY.md)
- [Flight adaptation and ROS recorder](docs/FLIGHT_ADAPTATION_QUICKSTART.md)
- [Publication preparation](docs/PUBLICATION.md)

The manifest explains which metadata/defaults differ from the development workstation. In particular, the smaller development batch is not the large-GPU comparison configuration. Do not treat the source bundle as a complete artifact reproducing a paper result.
