# Local Windows UI repair — 6 September 2026

This checkout is `C:\Users\wts28\Documents\PHD\particle_filter_cable_project`, running Windows 11 with an RTX 4080, Python 3.12.10 and PyTorch 2.11.0+cu128. Earlier packaging measurements in `LAB_VALIDATION.json` describe the RTX 5090 workstation; they are not measurements of this computer.

## Launcher

Use the PyCharm **Whip UI** configuration in `.run/Whip UI.run.xml`, or run `run_simulation.py` with this project's `.venv/Scripts/python.exe`. The working directory is the project root. `run_cable_twin.py` was retired. The old local run configuration was retargeted, with a backup at `.idea/workspace.xml.before-launcher-fix-20260906.bak`. PyCharm can retain configuration state until the project is reopened.

## PPO/SAC page failure

The user reported an `ImportError` during lazy construction of the PPO page's native 3D viewer:

```text
Failed to load vtkViewsContext2D: No module named vtkmodules.vtkViewsCore
```

Both native extension files are present in the installed VTK wheel. Fresh native GUI processes worked outside the failing PyCharm session, so an incomplete installation was not established and no package reinstall was performed.

`simulator/gui/viewer_3d.py` now explicitly imports `vtkmodules.vtkViewsCore`, then `vtkmodules.vtkViewsContext2D`, before constructing the PyVista Qt interactor. This avoids asking the context extension's native loader to resolve the missing dependency during lazy viewer initialization. The exact failure was not reproduced in fresh processes, so this is a targeted initialization fix validated by native desktop tests, not proof of the failing PyCharm session's underlying state.

The user requires the original PyVista/VTK desktop renderer. There is no automatic desktop fallback or swallowed renderer initialization error. The pre-existing offscreen-only renderer remains for headless UI tests. Both algorithms share the native initialization path. No reward, force timing, calibration, or saved policy parameters were changed.

## Verification

- Native Windows launcher regression (`tests/ui/test_native_viewer.py`): invoked `run_simulation.main()`, displayed the application, clicked repeatedly through PPO, SAC, Task & Rewards, Real-world Updates, and Data & Calibration; exercised every camera preset and scene updates; closed cleanly. Asserted both policy pages used `PointCableViewer3D` and had native render-window interactors. This ran outside PyCharm using the same project interpreter. Enable with `WHIP_TEST_NATIVE_GUI=1` on an interactive Windows desktop; the test starts a fresh process with the Windows Qt platform.
- Existing tests were corrected for the configured follow-through and 20 Hz query count. A temporary short-horizon PPO budget fixture now disables the incompatible one-second action prior only in its test configuration. A rollout-update fixture uses short recovery; separate tests cover recovery behavior.
- Targeted native launcher and exact PPO episode-budget tests: 2 passed. The two corrected point-force tests also passed their targeted run.
- `pip check`: no broken requirements. Verified installed wheel hashes for VTK (580 files), PyVista (203), and PyVistaQt (19): none missing or modified. Canonical active model hash matches the handoff.
- Final full suite on Windows/RTX 4080: **172 passed, 2 skipped, 4 warnings**, in 416.86 s. Native desktop regression was enabled. Warnings are PyTorch JIT deprecations in existing compilation tests. Command (PowerShell, project root):

  ```powershell
  $env:WHIP_TEST_NATIVE_GUI='1'
  .venv/Scripts/python.exe -c "import torch,pytest; torch.set_num_threads(1); raise SystemExit(pytest.main(['-q','--tb=short']))"
  ```

  Limiting CPU threads avoids oversubscription in the small simulation test batches; it does not change application configuration.

Installed graphics versions: PySide6 6.11.2, PyVista 0.48.4, PyVistaQt 0.12.0, VTK 9.6.2. These are observed local versions, not a statement that every combination or platform is validated.

## Remaining local assets

The selected PPO run and its checkpoint are still absent from this checkout. Navigation works without a trained run; running that selected policy requires the preserved private experiment bundle. No production PPO/SAC training or supervisor was restarted. Protected recording contents were not read. The canonical active model hash remains `e759615bacd8c075d0fa6318f881d1a870ba6d00a0bdbbaf174660fca704c1c2`.
