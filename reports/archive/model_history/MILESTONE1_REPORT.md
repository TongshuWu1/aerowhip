# Milestone 1 — Aerial Cable Simulation Foundation

## Outcome

Milestone 1 is complete. The repository root now contains a clean,
differentiable prescribed-root UAV/cable simulator. It reuses the existing
DDER mathematical implementation and deliberately excludes planning, fitting,
real-data ingestion, state estimation, and speculative physics.

## Directory tree

```text
run_simulator.py
README.md
requirements.txt
config/
  default.json
simulator/
  __init__.py
  simulator.py
  state.py
  parameters.py
  cable/
    __init__.py
    dder.py
    cuda_fixed_pcg.py
    config.py
    initialization.py
  uav/
    __init__.py
    state.py
    model.py
  coupling/
    __init__.py
    attachment.py
  gui/
    __init__.py
    app.py
    main_window.py
    viewer_3d.py
tests/
  __init__.py
  _common.py
  test_dder_static.py
  test_dder_rollout.py
  test_dder_gradients.py
  test_dder_regression.py
  test_simulator_api.py
offline_dder/                     # preserved offline EI/Cb workflow
legacy/current_baseline_2026-08-27/  # preserved previous project
```

## Reused numerical implementation

The following files were copied byte-for-byte:

- `offline_dder/cable_twin/shared/dder.py` -> `simulator/cable/dder.py`
- `offline_dder/cable_twin/shared/cuda_fixed_pcg.py` ->
  `simulator/cable/cuda_fixed_pcg.py`

SHA-256 equality was verified after all implementation work. Reused DDER
classes/functions include `DderModel`, `DderParameters`, `DderState`,
`START_PINNED_FREE_END`, nonlinear DER curvature/bending, implicit objective
Kelvin-Voigt damping, mass-weighted RATTLE position/velocity projection, and
the existing batched CUDA support.

## Refactoring around DDER

No mathematical DDER code was refactored. New code only provides:

- an explicit measured `CableConfiguration`;
- explicit tensor-capable `CableParameters(EI, Cb)`;
- a straight hanging-state initializer;
- minimal `UAVState`, `UAVCommand`, and `UAVCommandSequence` containers;
- the temporary `PrescribedRootModel`;
- an explicit one-way attachment boundary function; and
- `CoupledSimulator.reset`, `step`, and differentiable `rollout`.

The model object stores scalar reference EI/Cb only for fixed geometry and
stability metadata. Every DDER step receives the original EI/Cb tensors as
overrides, so rollout gradients remain connected to the caller's parameters.

## Numerical behavior

There is no intended numerical change to DDER. An independent 15-frame
one-attached rollout through the legacy and new module paths produced exactly
equal node positions, velocities, and constraint errors (`atol=0`, `rtol=0`).
The two numerical source files also remain byte-identical.

## Differentiability result

For a 24-frame aggressive X-reversal trajectory on CPU, the test loss was:

```text
loss = 0.002860981752804429
d(loss)/d(EI) = -0.07072711707495154
d(loss)/d(Cb) = -59.15771403679531
```

Both gradients are finite and nonzero. The core rollout contains no parameter
detachment; only GUI rendering detaches state for NumPy/Matplotlib display.

## Tests

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
```

Result:

```text
8 passed in 17.75 s
```

The tests cover static hanging behavior, edge constraints, deterministic
sinusoidal response, aggressive reversal, EI/Cb gradients, exact legacy
regression, stateful and stateless simulator APIs, and batched CUDA rollout.
Python compilation and `git diff --check` also pass.

## CPU/GPU status

- CPU float64 rollout: passed.
- CUDA batched rollout: passed.
- Detected GPU: NVIDIA GeForce RTX 4080.
- GUI production construction/reset: passed using the configured automatic
  device selection.

## GUI

Launch from the repository root:

```powershell
.\.venv\Scripts\python.exe run_simulator.py
```

The GUI displays the UAV/root, all 21 DDER nodes, ten measured-marker sites,
cable edges, and free tip. Controls are Start, Pause, Reset, Single Step,
editable EI/Cb, and the three prescribed-root debug modes: static, sinusoidal
X, and aggressive X reversal.

## Explicitly absent

No MPPI, system identification, OptiTrack loader, fitting GUI, observer,
receding-horizon control, detailed UAV dynamics, PID/thrust/rotor model, drag,
downwash, learned residual, reinforcement learning, or MuJoCo dependency was
introduced.

Development stops here pending the separately specified UAV-model milestone.
