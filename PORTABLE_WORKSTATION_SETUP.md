# Portable RTX 5090 workstation setup

This repository is prepared so a fresh clone contains the immutable production
model, selected PPO checkpoint, state banks, context normalizer, simulator UI,
and PyCharm run configurations. Raw experimental recordings and timestamped
training trees are deliberately not stored in Git.

## 1. Clone the correct branch

```powershell
git clone --branch twin-rewrite https://github.com/TongshuWu1/particle_filter_cable_project.git
cd particle_filter_cable_project
```

## 2. Install the NVIDIA driver and Python

Install a current NVIDIA Studio or Game Ready driver for the RTX 5090 and
64-bit Python 3.12. The project does not require a separately installed CUDA
Toolkit for normal PyTorch execution; the PyTorch wheel supplies its CUDA
runtime, while the NVIDIA display/compute driver remains required.

Create a project-local environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

The RTX 5090 is Blackwell (`sm_120`). Install a PyTorch build compiled with
CUDA 12.8 or newer. This command matches the tested environment used to prepare
the repository:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Do not install an older CUDA 12.1 PyTorch wheel for the 5090. `torch` is kept
out of `requirements.txt` intentionally so a generic dependency install cannot
silently replace the CUDA-enabled build with an unsuitable package.

## 3. Configure PyCharm once

1. Open the cloned repository folder as the PyCharm project.
2. Open **Settings → Project → Python Interpreter**.
3. Select **Existing environment** and choose
   `.venv\Scripts\python.exe` inside this clone.
4. Let PyCharm index the project.
5. Select **01 Workstation Preflight** in the run-configuration dropdown and
   press the green Run button.

The preflight must report `"status": "PASS"`. On the 5090 it should report a
CUDA capability of `[12, 0]` and include `sm_120` in the compiled architecture
list. It verifies all frozen artifact hashes and loads one finite deterministic
action from the selected PPO checkpoint. It never trains or accesses hardware.

## 4. Run with the PyCharm button

Committed configurations under `.run/` appear automatically:

- **01 Workstation Preflight** — read-only GPU/model/policy verification.
- **02 Simulator UI** — the complete PySide6/PyVista research UI.
- **03 PPO Replay - Selected D50** — one simulation-only replay.
- **04 PPO Training Preflight** — one small rollout/update integration test.
- **05 PPO Training - Portable Continuation** — immediately starts the
  configured 500,000-episode continuation; use deliberately.
- **06 Regression Tests** — the complete pytest suite.

Opening `run_simple_ppo.py` directly and pressing Run is also safe: with no
arguments it performs preflight rather than starting a long training job.
Normal training remains available from the UI or the explicit training run
configuration.

## 5. Portable and non-portable data

Included in Git:

- `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI` and its residual component;
- the small immutable fit/validation summaries required by the Model page;
- selected PPO checkpoints and configurations;
- production CEM reference data;
- training and validation state banks;
- publication plots and compact experiment histories.

Not included in Git:

- raw OptiTrack recordings;
- protected physical recordings;
- complete timestamped `data/policy_training` run directories;
- generated replay output and new training output.

Those optional local datasets can be copied into their existing `data/`
locations if the Experimental Data workspace is needed. Their absence does not
prevent the simulator, selected PPO replay, production model display, or PPO
training from running.

## 6. Safety and current scope

All supplied run configurations are simulation-only. The selected policy is a
10 Hz closed-loop nominal-physics PPO controller at the canonical target. It is
not authorized for hardware. The protected test remains not evaluated.

For the complete research handoff, read [`PPO_WHIP_HANDOFF.md`](PPO_WHIP_HANDOFF.md).
