# Install and launch

Use **Python 3.12** on Linux or Windows; the bootstrap also accepts Python 3.13.
The current direct-dependency versions are recorded in
[deployment constraints](../../requirements/deployment-constraints.txt).
These are tested local versions, not a transitive lockfile or a completed test
on every operating system.

## Ubuntu / NVIDIA workstation

Open a terminal in this repository and run:

```sh
python3 --version
python3 setup_lab.py --device cuda
sh start_lab.sh
```

If your default Python is older, invoke an installed Python 3.12 explicitly.
Python's venv module is required; on Ubuntu its package is normally
`python3-venv`. Use a graphical desktop session for the UI.

The installer creates a local environment, installs PyTorch 2.11.0 from the
official CUDA 12.8 wheel index, then installs the desktop dependencies.
The PyTorch project lists this [Linux/Windows installation combination](https://pytorch.org/get-started/previous-versions/).
Its [Blackwell support announcement](https://pytorch.org/blog/pytorch-2-7/)
explains why the CUDA 12.8 wheel family is appropriate for RTX 50-series hardware.
A working NVIDIA driver is still required on the host.

## Windows / NVIDIA workstation

```powershell
py -3.12 setup_lab.py --device cuda
.\start_lab.cmd
```

Paths containing spaces are supported. The launchers find their own repository
directory and do not depend on the terminal's current directory.

## Inspection on a computer without CUDA

```sh
python3 setup_lab.py --device cpu
sh start_lab.sh
```

The desktop, import and inspection functions can be used without a GPU.
The current full model-fitting implementation requires CUDA. The app must
report that restriction rather than silently use a different fitting method.

## Verify before the lab day

Linux:

```sh
.venv/bin/python tools/check_lab.py --compute --require-cuda --require-baseline
```

Windows:

```powershell
.venv\Scripts\python.exe tools/check_lab.py --compute --require-cuda --require-baseline
```

For a source-only checkout, first import the retained-M0 baseline bundle through
the app. The private handoff ZIP already includes it.

## If something needs attention

- **Wrong Python version:** create the environment using Python 3.12. Do not copy
  the developer's Windows environment onto Ubuntu.
- **CUDA unavailable:** inspect the health-check output and `nvidia-smi`. Check
  the installed driver and that the environment uses the CUDA wheel, not a CPU
  wheel. Use the [official PyTorch installation guide](https://pytorch.org/get-started/locally/)
  for your platform.
- **Qt cannot load xcb:** install the system libraries named in Qt's error output.
  On common Ubuntu installations, `libxcb-cursor0` and `libxkbcommon-x11-0`
  are frequent missing runtime dependencies. See the
  [Qt Linux requirements](https://doc.qt.io/qt-6/linux-requirements.html).
- **No display:** run in a desktop session. Remote/headless inspection uses
  `tools/lab.py` and `tools/check_lab.py`; no display is required for those.
- **Missing or changed baseline files:** re-import the original bundle into a
  fresh checkout. Do not edit hashes to make a modified asset pass.
- **Interrupted setup:** rerun the same setup command. It reuses the local
  environment; it does not delete experiment data.
- **Fit/planner job fails:** keep its log and input files. The log path is shown
  by the app. Do not replace a failed result with a historical model.

The installer downloads dependencies only when explicitly run. Opening the app
does not install software, fit a model or start a planner.
