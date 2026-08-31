# Milestone 2C — Fidelity Policy, Experimental I/O, and Scientific 3D Viewer

> **Geometry provenance notice (2026-08-27):** The I/O and viewer architecture
> in this report remains current, but its original geometry-dependent snapshots
> and hashes preceded the measured attachment update. The active simulator and
> viewer now use the 50-mm body-down connector offset and 60-mm
> connector-to-c1 interval documented in
> `MEASURED_GEOMETRY_UPDATE_REPORT.md`.

Date: 2026-08-27  
Git base: `cbdb59b` with the intentional in-progress project reorganization  
Hardware: NVIDIA GeForce RTX 4080  
Runtime: Python 3.12.10, PyTorch 2.11.0+cu128, CUDA 12.8  
Viewer: PySide6 6.11.2, PyVista 0.48.4, PyVistaQt 0.12.0, VTK 9.6.2

## Outcome

Milestone 2C is complete. The accepted Milestone 2B physical simulator was not
modified. The repository now has an explicit deterministic chain:

```text
FullStateCommand / FullStateCommandSequence
    -> CoupledSimulator
    -> SimulatorState / SimulatorTrajectory (complete latent state)
    -> OptiTrackObservation (pose + ten moving markers only)
```

`python run_simulator.py` launches one production PySide6/PyVistaQt
application. The former Tk canvas renderer, its orthographic projection math,
and all Tk imports have been removed from the active source. There is no second
production GUI or second physics implementation.

No fitting, MPPI, measurement noise, latency, dropout, drag, downwash,
residual dynamics, new UAV equations, or cable reaction into the UAV was
introduced.

## Architecture

Relevant current tree:

```text
run_simulator.py
config/
  default.json
simulator/
  simulator.py                 # only coupled physics implementation
  state.py                     # complete latent state/trajectory
  parameters.py
  observation/
    __init__.py
    optitrack.py                # deterministic observation projection only
  uav/
    model.py
    quaternion.py              # shared body-to-world quaternion convention
    state.py
  coupling/
    attachment.py
    root_boundary.py
  cable/
    dder.py
    cuda_fixed_pcg.py
  gui/
    app.py                      # QApplication and single entry path
    main_window.py              # controls and physics worker
    snapshot.py                 # compact immutable latest-state snapshot
    viewer_3d.py                # retained PyVista/VTK actors
tests/
  test_optitrack_observation.py
  test_gui_contract.py
MODEL_FIDELITY_POLICY.md
MILESTONE2C_REPORT.md
```

### Files added

- `simulator/observation/__init__.py`
- `simulator/observation/optitrack.py`
- `simulator/gui/snapshot.py`
- `tests/test_optitrack_observation.py`
- `tests/test_gui_contract.py`
- `MODEL_FIDELITY_POLICY.md`
- this report

### Files replaced or updated

- `simulator/gui/app.py`: Tk application replaced with `QApplication`.
- `simulator/gui/main_window.py`: Tk widgets replaced with Qt controls while
  retaining the independent physics worker and latest-state publication.
- `simulator/gui/viewer_3d.py`: custom canvas and projection math replaced by
  retained PyVista/VTK pipelines.
- `simulator/gui/__init__.py`: production GUI description updated.
- `simulator/__init__.py`: virtual OptiTrack API exported.
- `run_simulator.py`: remains the single launch command and now returns the Qt
  application exit code.
- `requirements.txt`: adds pinned Qt/PyVista/VTK versions and removes the old
  direct Matplotlib viewer dependency.
- `README.md`: records the experimental I/O and viewer semantics.

`CoupledSimulator`, the UAV models, DDER code, boundary code, parameters,
timestep, masses, discretization, and numerical backends were not edited.

## Model-fidelity policy

`MODEL_FIDELITY_POLICY.md` freezes the current research rule:

```text
MODEL FIDELITY FIRST
COMPUTATIONAL SPEED SECOND
```

The following remain part of one accepted production model:

- 21 DDER nodes and 20 edges;
- three DDER substeps per 10-ms physical frame;
- four constraint-projection iterations;
- measured cable/marker mass and interval distribution;
- rigid centerline clamp through DDER nodes 0 and 1;
- current `EI`, `Cb`, FullState UAV response, and CUDA runtime.

CUDA parallelism, batching, allocation/synchronization cleanup, and algebraically
equivalent acceleration remain implementation work. Reducing nodes, substeps,
projection iterations, damping fidelity, constraint enforcement, or timestep
is explicitly a new model requiring scientific validation.

## Experimental I/O semantics

The three concepts are now represented by separate types/modules.

### Input

```text
p_cmd
v_cmd
a_cmd
q_cmd       (xyzw)
omega_cmd
```

These remain FullState controller inputs. They are not labeled as measured
physical state.

### Internal UAV and cable state

```text
UAV:
  position
  velocity
  orientation quaternion
  angular velocity

cable:
  all 21 DDER positions
  all 21 DDER velocities
```

The latent state remains available in `SimulatorState` and
`SimulatorTrajectory` for propagation, differentiation, and evaluator-only
diagnostics.

### Virtual OptiTrack output

`OptiTrackObservation` contains exactly:

```text
uav_position_m                 [..., 3]
uav_orientation_xyzw           [..., 4]
cable_marker_positions_m       [..., 10, 3]
```

The cable marker map is exactly:

```text
[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
```

Node 0 and the ten unmarked latent nodes are not exposed as moving OptiTrack
markers. UAV velocity, UAV acceleration, UAV angular velocity, cable velocity,
forces, and solver state are absent from the observation type.

The extraction functions support both batched states and time-major batched
trajectories. They create observation-owned tensors, do not mutate or alias
mutable latent state storage, and preserve differentiability. Repeated
extraction is deterministic. No measurement noise or camera model is present.

## Physics/render separation

The execution architecture is:

```text
physics worker
    -> production CoupledSimulator.step()
    -> compact immutable VisualizationSnapshot (latest value only)
    -> Qt render timer
    -> retained VTK point/actor updates
```

The worker owns physics advancement. The Qt thread owns VTK. No CUDA DDER step
runs in a render callback, and no GUI class can write simulator state. Compact
snapshots contain only:

- simulated UAV position/quaternion;
- FullState commanded position/quaternion when available;
- simulator boundary attachment position;
- 21 cable positions;
- two scalar status diagnostics.

They do not contain the autograd graph, solver workspaces, complete velocity
arrays, acceleration, angular velocity, or gradients. The worker publishes up
to 50 compact snapshots/s; the approximately 30-Hz renderer consumes only the
newest snapshot. No stale-frame queue exists.

PyVistaQt's separate default 5-Hz automatic redraw timer was disabled. The
application has one explicit render cadence, avoiding duplicate GPU work and
preserving CUDA physics throughput.

## Viewer

### UAV

The simulated UAV is a simplified Crazyflie-like mesh made from a central
body, four diagonal arms, four rotor disks, and a contrasting +body-X nose.
The asymmetry makes roll, pitch, and yaw visually legible.

Every simulated UAV actor uses one homogeneous body-to-world transform from:

```text
p_uav_sim, q_uav_sim
```

The rotation is computed by the same
`quaternion_to_rotation_matrix_xyzw()` function used by the physical rigid
attachment. No graphics-only Euler state exists. Identity and +90-degree roll,
pitch, and yaw cases are covered by tests.

An optional translucent commanded-pose ghost uses only `p_cmd` and `q_cmd`.
Its architecture is separate from the simulated actor and leaves an explicit
future slot for a measured-pose actor, but no real-data loading was added.

An optional body XYZ triad receives the identical simulated pose transform.

### Attachment and cable

- The purple attachment actor uses the exact attachment position returned by
  the simulator boundary evaluation.
- The visible first edge is the actual DDER segment `r_0 -> r_1`.
- A persistent `vtkPolyLine -> vtkTubeFilter` pipeline renders all 21 DDER
  nodes.
- Physical cable radius is 1.75 mm. The default graphics-only radius scale is
  2.0 for visibility; it changes no DDER geometry or parameter.
- Nine orange spheres and one larger red free-tip sphere show all ten virtual
  OptiTrack material sites.

### Scene and camera

The restrained scene includes a ground plane, grid, scale labels, world axes,
smooth materials, key/fill lighting, and perspective projection. Native VTK
interaction provides orbit, pan, and zoom. Presets provide Perspective, Front,
Side, Top, and Follow UAV. Follow mode translates the camera target while
preserving the current camera offset.

No velocity, acceleration, angular-velocity, torque, thrust, force, or gradient
actors are shown by default.

### Controls

The Qt control panel retains:

- Start, Pause, Reset, Single Step;
- Prescribed Root / Pivot and FullState UAV / Clamped;
- Hover, X Sinusoid, Aggressive X Reversal, Small Attitude Motion;
- `K_p`, `K_v`, `k_a`, `K_R`, `K_omega`, `EI`, and `Cb`;
- concise `NOT IDENTIFIED` labeling for the UAV response parameters;
- commanded-pose, body-frame, and camera controls.

## Performance

Measured on the RTX 4080 in a warmed 10-s interactive Aggressive X Reversal
run using the production 21-node clamped CUDA simulator:

| Quantity | Mean/effective | p95 |
|---|---:|---:|
| physics worker | **98.21 Hz** | — |
| VTK render | **29.62 Hz** | — |
| snapshot pack/synchronize/transfer | **1.791 ms** | **3.304 ms** |
| retained renderer update | **3.844 ms** | **5.068 ms** |

The snapshot measurement is end-to-end and includes the CUDA completion
synchronization encountered by the compact device-to-host transfer. Physics
and rendering use independent clocks; the configured physical timestep remains
10 ms and is never changed by renderer load.

A 40-Hz graphics cadence was rejected because the discrete GPU contention
measurably reduced physics throughput. The approximately 30-Hz retained VTK
cadence is therefore the fidelity-first default on this hardware.

## Verification

### Observation and pose contracts

Tests verify:

- UAV observation position/orientation equal simulator pose exactly;
- marker shape is exactly `[..., 10, 3]`;
- marker values are exactly nodes 2,4,...,20;
- latent nodes and all velocity channels are absent;
- extraction is deterministic, batched, non-mutating, and storage-isolated;
- identity and +90-degree roll/pitch/yaw viewer transforms;
- snapshot attachment equals the physical DDER root;
- displayed first two cable points are the actual clamped DDER nodes;
- graphics snapshots are immutable and omit latent velocity arrays.

The existing attitude-motion regression also verifies that changing simulated
UAV orientation changes the physical root tangent and cable motion, rather
than only changing graphics.

### Physics trajectory regression

Before GUI/observation changes, a deterministic float64 30-step coupled
translation-plus-attitude rollout was hashed field by field. After all changes,
every SHA-256 value remained exactly identical:

| Tensor | Pre/post SHA-256 |
|---|---|
| UAV position | `704beb56c759e2c4fa801d892b830db1ed03ba3ff694534f2f8088a2947b262d` |
| UAV velocity | `2cb37325fcef543157b301864e3115b3f44fc690da733d16dfbb1fd8c31979d2` |
| UAV orientation | `858042a2216d2eae96d7965d28c108b9b156df116648064617fbd29a2114557d` |
| UAV angular velocity | `c94be1ee5c8bf93c2228c32183967af34a5f8e16353790adf6e499f25b67bb8a` |
| attachment position | `704beb56c759e2c4fa801d892b830db1ed03ba3ff694534f2f8088a2947b262d` |
| cable position | `51c5c49e0bc21617355ee9de485c115070d164b3e82c30892923e68d9a02638a` |
| cable velocity | `584f76bf11b3fe4b65dd3a9e5be0e489d31af8ef5b9927a7de27fdd0ddf20f45` |

This is exact byte equality, not tolerance-based similarity.

The complete active suite passes:

```text
28 passed
```

It includes DDER static/dynamic/gradient tests, FullState and clamped-boundary
tests, CUDA production-runtime tests, simulator API tests, observation tests,
and GUI data-contract tests.

### Windows VTK loading

Windows Application Control blocked VTK's optional NetCDF DLL when the
monolithic `import vtk` attempted to load every VTK component. The viewer uses
VTK's supported modular imports for only common data, geometry filters,
sources, and rendering. NetCDF is not required by this renderer. The actual Qt
GUI launch and live OpenGL scene were tested successfully.

## Cleanup and scope confirmation

- The old Tk canvas renderer and custom projection math are gone.
- No `tkinter`, Tk widget, or old orthographic-view code remains under
  `simulator/`.
- There is one GUI entry point: `python run_simulator.py`.
- There is one physical simulator: `CoupledSimulator`.
- Observation code is independent of GUI code and contains no physics.
- Physics was not moved into VTK/PyVista.
- No noise, fitting, MPPI, real-data loader, synchronization pipeline, drag,
  downwash, residual model, uncertainty model, firmware emulation, or new
  coupling was added.

Milestone 2C stops here. The real-data loader and fitting pipeline have not
been started.
