# Maintained checks

Tests are grouped by the behavior they protect. Tests exclusively exercising retired SAC/PPO trainers and policy launchers were removed with those features. Shared physics, fitting, MPPI, command correction, recovery, export and UI checks remain. Mixed test files retain their non-policy checks.

| Group | Coverage |
|---|---|
| `physics/` | DDER, mass/force accounting, damping, GPU parity and reward environment |
| `calibration/` | Recorded-data contracts, initialization, physical fitting and derivatives |
| `training/` | MPPI objectives, open-loop simulation, progress, numerical parity and release packaging |
| `flight/` | Single execution, PID recovery, replay, flight imports and adaptation |
| `ui/` | Current research workflow, controls, plotting and validation preview |

Run `.venv/Scripts/python.exe run_tests.py <group>` from the project root, or omit the group for the full suite. Multiple groups are supported. Individual tests can be selected with ordinary pytest paths, for example:

```powershell
.venv/Scripts/python.exe -m pytest tests/training/test_legacy_retirement.py -q
```

Folders under `archive/`, `results/`, `runs/`, and `data/` are not test-discovery locations. They can contain frozen historical source that must not be collected as additional tests. Protected real recordings remain outside development tests; keep their exclusion rules intact.
