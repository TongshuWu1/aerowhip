# Milestone 2B — Production Clamped DDER Runtime

> **Geometry provenance notice (2026-08-27):** This milestone validated the
> production runtime before the measured attachment update. Its original
> geometry-dependent hashes and numerical trajectories are historical. The
> active simulator now uses the measured 50-mm body-down connector offset and
> 60-mm connector-to-c1 interval documented in
> `MEASURED_GEOMETRY_UPDATE_REPORT.md`; the runtime method itself is unchanged.

Date: 2026-08-27  
Git base: `cbdb59b` with the intentional in-progress project reorganization  
Hardware: NVIDIA GeForce RTX 4080, driver 610.88, 16,376 MiB  
Runtime: Python 3.12.10, PyTorch 2.11.0+cu128, CUDA 12.8

## Outcome

The 21-node FullState/clamped simulator now uses the generalized production
CUDA runtime during normal execution. The public research API remains:

```python
simulator.step(command, parameters)
simulator.rollout(initial_state, commands, parameters)
```

No cable equation, physical parameter, timestep, substep, constraint
projection count, node count, mass, or attachment geometry was changed.

Warmed batch-one complete simulation decreased from approximately 142 ms per
10-ms frame to **6.451 ms mean**. The interactive GUI advances simulation at
**99.88 Hz** while rendering at **21.64 Hz** using the same production
`CoupledSimulator`.

## Architecture

### Files changed

- `simulator/cable/cuda_fixed_pcg.py`
- `simulator/cable/dder.py`
- `simulator/simulator.py`
- `simulator/uav/model.py`
- `simulator/gui/main_window.py`
- `simulator/gui/viewer_3d.py`
- `tests/test_clamped_production_runtime.py`
- `tests/test_dder_regression.py`
- `pytest.ini`
- `README.md`
- this report

### Previous limitation

The old fused runtime assumed exactly one prescribed node:

- free coordinates were hard-coded as nodes 1 through N-1;
- the damping vector dimension was `3*(N-1)`;
- only one boundary velocity was accepted;
- damping input/output offsets were hard-coded to three scalars;
- projection assigned zero inverse mass only to node 0;
- boundary replacement restored only node 0;
- CUDA launch validation rejected a two-node boundary;
- `DderModel.step_runtime()` enabled fused damping/projection only for
  `START_PINNED_FREE_END`.

The `ctypes` NVRTC launches also had no PyTorch autograd registration. The
runtime bending-force helper detached state and evaluated its energy gradient
with `create_graph=False`, so the old inference path was not a valid fitting
path.

### Topology generalization

One generated CUDA mechanics implementation now accepts a prescribed-start
count of one or two. For the clamped 21-node system:

- prescribed nodes: 0 and 1;
- free nodes: 2 through 20;
- free damping dimension: 57;
- curvature residual dimension: 57;
- boundary velocity shape: `B x 2 x 3`;
- node 0 and node 1 inverse masses are zero during projection;
- edge 0 remains in the constraint system but has two prescribed endpoints;
- edge 1 provides the first fixed/free coupling;
- all prescribed positions and velocities are restored after projection.

The already validated pivot CUDA source string remains byte-identical. Pivot
and clamp use the same source generator, launch class, DDER step, curvature,
damping, and projection definitions. There is no `fast_clamped_step()` or
copied interior model.

### Production precision

Normal CUDA construction selects float32 state, force, geometry, and
projection arithmetic. The fused damping kernel assembles its local operators
from float32 state but retains the established float64 PCG recurrence, then
returns float32 velocity. CPU and explicitly requested float64 execution retain
the high-precision PyTorch reference for regression.

This choice was validated against float64 in physical units below. No automatic
mixed precision was added elsewhere.

### Autograd strategy

Both inference and fitting call `DderModel.step_runtime()` through the same
`CoupledSimulator` state transition.

- no-grad CUDA execution uses fused PCG32 damping and fused four-position plus
  one-velocity projection;
- gradient execution uses ordinary differentiable PyTorch operations for the
  same curvature, damping, integration, and projection equations;
- gradient damping uses batched Cholesky on the exact same SPD Kelvin–Voigt
  system rather than recording 32 explicit PCG recurrences;
- bending-force evaluation now retains the requested higher-order graph so EI,
  Cb, UAV state, boundary pose, and all five UAV response parameters remain
  connected.

The direct gradient solve and PCG32 inference solve are numerical executions of
the same implemented implicit equation. Their complete trajectory and gradient
agreement was measured rather than assumed.

No CUDA graph capture was added. The fused operators launch on the active
PyTorch CUDA stream and remain compatible with future capture work, but current
performance already reaches the configured batch-one physics budget.

## Numerical equivalence

The production float32 CUDA simulator was compared with the generic float64
reference for 60 frames (0.6 s) in each condition. Values below are maximums
over every frame.

| Motion | all-node position | measured-site position | free-tip position | velocity | edge-length difference | root-tangent vector |
|---|---:|---:|---:|---:|---:|---:|
| Static hang | 0.088 µm | 0.073 µm | 0.055 µm | 8.79e-10 m/s | 0.100 µm | 0 |
| X sinusoid | 2.113 µm | 2.113 µm | 0.915 µm | 2.58e-5 m/s | 0.174 µm | 0 |
| Aggressive reversal | 2.189 µm | 2.189 µm | 1.852 µm | 8.00e-5 m/s | 0.156 µm | 0 |
| Attitude motion | 0.156 µm | 0.146 µm | 0.085 µm | 1.04e-5 m/s | 0.204 µm | 2.92e-7 |
| Translation + attitude | **2.766 µm** | **2.766 µm** | **1.887 µm** | **8.56e-5 m/s** | 0.174 µm | 5.70e-7 |

The informative trajectory loss differed by 8.55e-7 relative. These errors are
orders of magnitude below millimetre-scale motion-capture and cable-model
uncertainty.

The clamp invariants are restored every substep. The retained CUDA regression
requires nodes 0 and 1 to agree with the rigid attachment boundary within
2e-7 m in float32.

## Gradient equivalence

An informative 16-frame combined translation/attitude rollout was differentiated
with respect to all seven parameters. Reference is CPU float64; production is
CUDA float32 through the production runtime.

| Parameter | Reference gradient | Production gradient | Absolute difference | Relative difference | Sign |
|---|---:|---:|---:|---:|---|
| K_p | 3.55419585e-6 | 3.55424095e-6 | 4.51e-11 | 1.27e-5 | same |
| K_v | 4.97922144e-5 | 4.97924266e-5 | 2.12e-10 | 4.26e-6 | same |
| k_a | -2.32224051e-4 | -2.32227219e-4 | 3.17e-9 | 1.36e-5 | same |
| K_R | 8.86041366e-8 | 8.86039686e-8 | 1.68e-13 | 1.90e-6 | same |
| K_omega | 1.35620117e-6 | 1.35620212e-6 | 9.54e-13 | 7.03e-7 | same |
| EI | 6.70678690e-2 | 6.70684129e-2 | 5.44e-7 | 8.11e-6 | same |
| Cb | 2.65790482 | 2.65802574 | 1.21e-4 | **4.55e-5** | same |

All gradients are finite, nonzero under an informative excitation, and agree
in sign. The worst relative difference is 4.55e-5 for Cb.

The differentiable path is throughput-oriented rather than interactive:
one forward-plus-backward frame measured approximately 381 ms at B=1 and
382 ms at B=256, corresponding to approximately 2.6 and 670 differentiated
trajectory-steps/s respectively. Future fitting should batch windows and
candidate parameters. This is not presented as a real-time gradient loop.

## Performance

All timings are warmed, 30 synchronized measurements, unchanged 21-node cable,
3 substeps, 4 position projection iterations, 10-ms physical frame, RTX 4080.
One-time NVRTC compilation is excluded.

| Batch | DDER mean ms/frame | Complete simulator mean | Median | p95 | trajectories/s | peak PyTorch memory |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.966 | **6.451 ms** | 6.388 | 7.928 | 155 | 8.15 MiB |
| 16 | 4.147 | 7.002 ms | 6.700 | 9.793 | 2,285 | 8.22 MiB |
| 64 | 4.143 | 6.368 ms | 6.253 | 8.027 | 10,050 | 8.44 MiB |
| 256 | 4.269 | 6.389 ms | 6.037 | 8.357 | **40,067** | 9.34 MiB |

Separately measured batch-one UAV response plus rigid attachment cost was
1.930 ms mean. Component and complete timings were independent samples, so
their percentiles should not be algebraically combined.

### GUI

The original Matplotlib renderer cost approximately 57.3 ms per frame and held
Python's GIL. Updating artists in place was already being done, so Matplotlib
3D was replaced with a retained Tk canvas using a fixed orthographic 3D camera.
This changes visualization only.

| Quantity | Result |
|---|---:|
| compact CUDA-to-CPU visualization transfer | 0.151 ms mean |
| retained viewer update | 0.726 ms mean, 1.109 ms p95 |
| configured physics rate | 100 Hz |
| measured GUI physics rate | **99.88 Hz** |
| measured render rate | **21.64 Hz** |
| simulation-time / wall-time | **0.999** |

Physics runs in a dedicated worker and publishes immutable latest-state
snapshots. Rendering reads one compact detached snapshot containing cable
positions, UAV pose, command marker, tip speed, and edge error. Rendering never
owns or modifies physics state.

## Regression

- current root suite: **20 passed**;
- legacy pivot generic rollout remains bitwise identical;
- legacy pivot CUDA source remains byte-identical;
- normal clamped CUDA execution is asserted to report
  `production_cuda_fused`;
- fused clamped PCG32/projection agrees with the differentiable PyTorch PCG32
  execution within 1e-6 m position and 7e-5 m/s velocity in the retained test;
- all seven production parameter-gradient paths are asserted finite;
- batched CUDA rollout passes;
- clamped rigid-boundary geometry passes;
- Python compilation passes.

`pytest.ini` limits the active root suite to `tests/`; archived legacy tests and
the separately preserved `offline_dder` project are not accidentally collected
as current Milestone 2 tests.

## Cleanup and scope

- no temporary benchmark scripts were retained;
- no backend selector or benchmark controls were added to the GUI;
- no duplicate GUI/fitting/MPPI cable model was introduced;
- no MPPI, fitting, real-data loader, noise, drag, downwash, residual model, or
  new coupling physics was added;
- no temporary CPU GUI workaround remains;
- normal GUI CUDA execution uses the production clamped backend and displays
  its execution status for diagnosis only.

Milestone 2B stops here. Real-data parameter fitting has not been started.
