# Installation

For the current private lab bundles and PyCharm/Ubuntu instructions, start with
[lab setup](LAB_SETUP.md). The selected-policy bundle differs from a source-only release.

The development workstation uses Windows and Python 3.12. The tested direct-package inventory is in `requirements/environment.windows-py312.json`. It records one observed environment, not a complete cross-platform lockfile. Hardware deployment and ROS are not part of this installer.

## Create an isolated environment

From the extracted source directory on Windows:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

For headless computation without the desktop dependencies, install `requirements/headless.txt` instead. Python's [venv documentation](https://docs.python.org/3/library/venv.html) describes virtual environment creation and activation; the commands above use the environment's interpreter directly.

For NVIDIA training, choose the wheel appropriate to the machine using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Do not assume an installation has GPU support solely because a GPU is present. The recorded development wheel is `2.11.0+cu128`; its exact availability and platform compatibility must be checked when recreating the environment.

```powershell
.venv/Scripts/python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

No packages are downloaded or installed by launching the simulator itself.

## First run

```powershell
.venv/Scripts/python.exe run_simulation.py --headless --device cpu --duration 0.05
.venv/Scripts/python.exe run_simulation.py
```

The first command performs a short hanging-hover simulation and reports numerical status. The second opens the desktop interface. A source-only release has an empty recording table and no trained checkpoints; these are expected states.

Run the self-contained smoke tests with:

```powershell
.venv/Scripts/python.exe -m pytest tests/physics/test_point_mass.py tests/training/test_early_stop_comparison.py tests/flight/test_flight_adaptation.py -q
```

The full development suite also includes GPU tests and checks tied to separately held experimental recordings. Those checks do not establish fresh-install support when the data are absent. See [reproducibility](REPRODUCIBILITY.md). The GitHub workflow runs the CPU smoke subset; adding its YAML does not mean a remote CI run has occurred.
