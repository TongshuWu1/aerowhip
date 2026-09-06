# Maintained checks

Tests are grouped by the current behavior they protect. Tests that only exercised superseded UI pages, MPCC execution or a frozen historical fit have been removed. Physics, current fitting, PPO/SAC execution, stop/checkpoint integrity and current UI regressions remain.

| Group | Coverage |
|---|---|
| `physics/` | DDER, mass/force accounting, damping, GPU parity and reward environment |
| `calibration/` | Recorded-data contracts, initialization, physical fitting and derivatives |
| `training/` | PPO/SAC, checkpoints, open-loop collection, progress and stopping |
| `flight/` | Single execution, PID recovery, replay, flight imports and adaptation |
| `ui/` | Current research workflow, controls, plotting and validation preview |

Run `.venv/Scripts/python.exe run_tests.py <group>` from the project root, or omit the group for the full suite. Multiple groups are supported. Individual tests can be selected with ordinary pytest paths, for example:

```powershell
.venv/Scripts/python.exe -m pytest tests/training/test_early_stop_comparison.py -q
```

Folders under `archive/`, `results/`, `runs/`, and `data/` are not test-discovery locations. They can contain frozen historical source that must not be collected as additional tests. Protected real recordings remain outside development tests; keep their exclusion rules intact.
