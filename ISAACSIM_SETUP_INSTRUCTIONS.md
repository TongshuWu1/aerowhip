# Isaac Sim and Isaac Lab setup instructions for the RTX 5090 computer

## Instruction to Codex on the new computer

Read this entire file before taking action, then carry out the installation and
verification from beginning to end. You are authorized to inspect the computer,
clone the project, create the Python environment, install the pinned packages,
apply the included compatibility patch, and run the tests described below.

Do not redesign or upgrade the project's physics stack. Do not overwrite an
existing repository or environment without inspecting it first. Ask the user
only when NVIDIA requires the user to accept the EULA, a driver installation
requires approval or a reboot, credentials are needed, or a genuine blocker
cannot be resolved safely. Continue autonomously through ordinary installation
and troubleshooting steps, and finish by giving the user the complete report
listed at the end of this file.

The goal is to reproduce the project's Isaac drone-cable plant on the second
Windows computer with an RTX 5090.

## Fixed research stack

Keep both computers on the same versions:

- Windows 11 x64;
- Python 3.11;
- Isaac Sim 5.1.0;
- Isaac Lab tag `v2.3.2`, commit
  `37ddf626871758333d6ed89cf64ad702aef127d0`;
- PyTorch 2.7.0 with CUDA 12.8;
- Torchvision 0.22.0 with CUDA 12.8;
- HDF5 binding `h5py==3.15.1`; and
- the canonical cable artifact
  `optitrack_offline/models/cable_model.json`, SHA-256
  `0eece55cf9c2b07fcd699c1e7d9f5079d01240aabda2a7237a54b8c3c4fbc827`.

Do not install the current Isaac Lab `develop` branch or Python 3.12. Do not
upgrade only one workstation after experiments begin. A separate CUDA Toolkit
is not required; the PyTorch wheel supplies its CUDA 12.8 runtime.

Official references:

- [Isaac Lab v2.3.2 pip installation](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/pip_installation.html)
- [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [Current Omniverse driver matrix](https://docs.omniverse.nvidia.com/dev-overview/latest/common/technical-requirements.html)
- [PyTorch 2.7 Blackwell support](https://pytorch.org/blog/pytorch-2-7/)
- [PyTorch 2.7 CUDA 12.8 wheels](https://pytorch.org/get-started/previous-versions/)

## Rules for the setup agent

1. Inspect before changing anything. Record Windows, GPU, driver, Python,
   repository commit, and free disk space.
2. Do not delete, overwrite, or reset an existing project or Isaac Lab clone.
   If a target directory already exists, inspect it and use a fast-forward pull
   only when safe.
3. Do not accept the NVIDIA EULA on the user's behalf. Stop at the first EULA
   prompt and ask the user to accept or reject it.
4. Do not copy the primary RTX 4080 computer's D3D12 workaround by default.
   Test the RTX 5090 with Isaac Lab's default Vulkan configuration first.
5. Do not commit generated USD, run reports, OptiTrack CSVs, videos, or policy
   checkpoints. They are intentionally ignored.
6. Do not load old SAC checkpoints into this new plant. Their action space and
   transition model are incompatible.

## 1. Inspect the second computer

Run from PowerShell:

```powershell
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
Get-ComputerInfo | Select-Object WindowsProductName,WindowsVersion,OsBuildNumber,CsTotalPhysicalMemory
Get-PSDrive -PSProvider FileSystem | Select-Object Name,Free,Used
py -3.11 --version
git --version
```

The RTX 5090 should report about 32 GB VRAM. Use an official NVIDIA
Production/Studio driver selected for the RTX 5090. NVIDIA's current validated
Blackwell Windows matrix lists `595.97` WHQL; on fully patched Windows 11,
Vulkan on the R580 branch requires at least `582.41`. If the selector now
offers a newer driver, record its exact version and validate it below. Driver
installation may require a reboot, so report that before initiating it.

Enable Git long paths:

```powershell
git config --global core.longpaths true
```

## 2. Clone the project branch

Choose a parent directory with ample free space. The commands below use
`%USERPROFILE%\Documents\PHD`:

```powershell
$projectParent = Join-Path $env:USERPROFILE "Documents\PHD"
$project = Join-Path $projectParent "particle_filter_cable_project"
New-Item -ItemType Directory -Force -Path $projectParent | Out-Null
git clone --branch twin-rewrite --single-branch https://github.com/TongshuWu1/particle_filter_cable_project.git $project
Set-Location $project
git status --short
git log -1 --oneline
```

If `$project` already exists, do not clone over it. Inspect its remote, branch,
status, and local changes first.

## 3. Create the Python 3.11 environment

PowerShell script activation may be disabled. Activation is not required;
invoke the environment's Python by absolute path:

```powershell
$isaacEnv = Join-Path $env:USERPROFILE "env_isaaclab"
py -3.11 -m venv $isaacEnv
$isaacPython = Join-Path $isaacEnv "Scripts\python.exe"
& $isaacPython --version
& $isaacPython -m pip install --upgrade pip setuptools
```

Install the pinned simulator and Blackwell-capable PyTorch wheels:

```powershell
& $isaacPython -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
& $isaacPython -m pip install --upgrade torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

The first Isaac Sim launch downloads extensions and shaders and can legitimately
take ten minutes. It also presents NVIDIA's EULA; ask the user to respond.

Verify Isaac Sim before installing Isaac Lab:

```powershell
$isaacSim = Join-Path $isaacEnv "Scripts\isaacsim.exe"
& $isaacSim isaacsim.exp.compatibility_check
& $isaacSim isaacsim.exp.full.kit
```

## 4. Clone and patch the exact Isaac Lab release

Keep the Isaac Lab checkout outside the project repository:

```powershell
$lab = Join-Path $projectParent "IsaacLab"
git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git $lab
git -C $lab rev-parse HEAD
git -C $lab describe --tags --exact-match HEAD
```

The outputs must be the exact commit and tag listed above. Apply the repository's
small FastAPI/Starlette compatibility patch, then inspect the diff:

```powershell
$starlettePatch = Join-Path $project "isaac_whip\patches\isaaclab_v2.3.2_starlette.patch"
git -C $lab apply --check $starlettePatch
git -C $lab apply $starlettePatch
git -C $lab diff -- source/isaaclab/setup.py
```

Install Isaac Lab without third-party RL frameworks because this project has
its own SAC implementation. Use `activate.bat` through `cmd.exe` to ensure the
batch installer sees the virtual environment:

```powershell
$installCommand = "call `"$isaacEnv\Scripts\activate.bat`" && cd /d `"$lab`" && call isaaclab.bat -i none"
cmd.exe /d /c $installCommand
```

Pin the two Windows ABI/package details already validated on the primary PC:

```powershell
& $isaacPython -m pip install h5py==3.15.1 wheel==0.45.1
& $isaacPython -m pip check
```

`pip check` must report `No broken requirements found`. Do not upgrade
`packaging`; Isaac Lab 2.3.2 installs `packaging==23.0`.

## 5. Verify RTX 5090 CUDA and installed versions

```powershell
& $isaacPython -c "from importlib.metadata import version; print('isaacsim',version('isaacsim')); print('isaaclab-package',version('isaaclab')); print('torch',version('torch')); print('torchvision',version('torchvision')); print('starlette',version('starlette')); print('packaging',version('packaging')); print('h5py',version('h5py'))"

& $isaacPython -c "import torch; print('device=',torch.cuda.get_device_name(0)); print('capability=',torch.cuda.get_device_capability(0)); print('compiled_arches=',torch.cuda.get_arch_list()); print('cuda_runtime=',torch.version.cuda); print('available=',torch.cuda.is_available()); x=torch.randn((4096,4096),device='cuda'); print('cuda_test=',float((x@x.T).mean()))"
```

Required results:

- device name contains `RTX 5090`;
- capability is `(12, 0)`;
- compiled architectures include `sm_120`;
- CUDA runtime is `12.8`;
- CUDA is available; and
- the matrix operation completes without `no kernel image`, access violation,
  or a frozen process.

The editable Isaac Lab package reports its internal Python-package version
`0.54.2`; use the Git tag and commit—not that package number—as the framework
release identity.

## 6. Verify Isaac Lab and the project

```powershell
$tutorial = Join-Path $lab "scripts\tutorials\00_sim\create_empty.py"
& $isaacPython $tutorial --headless
& $isaacPython $tutorial
```

Then run the project's pure tests and finite physics smoke checks:

```powershell
Set-Location $project
& $isaacPython -m unittest discover -s tests -p "test_isaac_whip*.py" -v

.\run_isaac_whip.bat --headless --num-envs 1 --link-count 20 --mode hover --duration-s 0.5 --output-json data\isaac_whip\rtx5090_hover.json
.\run_isaac_whip.bat --headless --num-envs 64 --link-count 20 --mode excite --duration-s 0.5 --output-json data\isaac_whip\rtx5090_64env.json
.\run_isaac_whip.bat --num-envs 1 --link-count 20 --mode excite --duration-s 10 --output-json data\isaac_whip\rtx5090_gui.json
```

Each project report must say `PASS`, remain finite, and pass its authored-USD,
resolved-body-property, topology, and constraint checks. The passive-joint
warning (`0 != 57` Isaac Lab actuators at 20 links) is expected because the
bending drives are authored directly in PhysX.

If the environment is installed somewhere else, set a persistent or temporary
override before using the batch launcher:

```powershell
$env:ISAAC_WHIP_PYTHON = "D:\path\to\env_isaaclab\Scripts\python.exe"
```

## 7. Graphics-backend troubleshooting

Keep Vulkan as the nominal RTX 5090 baseline. If the GUI crashes specifically
inside `rtx.scenedb`, test D3D12 for one run without modifying Isaac Lab:

```powershell
& $isaacSim isaacsim.exp.full.kit --/app/vulkan=false
```

Only make D3D12 persistent if the same driver repeatedly fails Vulkan and the
D3D12 diagnostic succeeds. Record the GPU, driver, Windows build, backend, and
exact failure in the project notes. Do not describe a backend workaround as a
change to the physical model.

## Completion report expected from Codex

Return all of the following to the user:

1. project branch and commit;
2. Isaac Lab tag and commit;
3. Python, Isaac Sim, PyTorch, CUDA runtime, Starlette, h5py, and driver versions;
4. RTX 5090 capability and compiled architecture list;
5. `pip check` result;
6. headless tutorial result;
7. one-environment, 64-environment, and GUI project report statuses;
8. graphics backend used; and
9. every local compatibility patch applied.
