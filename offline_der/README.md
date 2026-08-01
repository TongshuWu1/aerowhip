# Offline Free-Cable DDER Identification

This program identifies the free-cable mechanics used later by the online
DDER particle filter. It is system identification, not neural-network training.
It currently estimates only bending stiffness and effective velocity damping.
Cube contact, friction, force, and online particle filtering are deliberately
absent.

## Dedicated offline UI

Launch the offline application with:

```powershell
.\.venv\Scripts\python.exe run_offline_der.py
```

The application provides one ordered workflow:

1. acquire a free-cable motion while watching the canonical 3D and PIDNet
   segmentation viewer;
2. extract measured reference trajectories from the completed lossless SVO;
3. identify and save the deployment DDER model.

The controller launches the existing `run_cable_twin.py` viewer in a separate
process; it does not duplicate the camera, perception, graph, or OpenGL code.
Start/stop controls operate the same lossless SVO recorder as the viewer's `R`
key, and each finalized recording is added automatically to the extraction
page. Close the live viewer before extraction or fitting so the offline process
has exclusive use of the camera and CUDA resources. Extraction quality gates
and the important fitting settings remain editable in the window.

The fitting page takes measured total cable mass and computes linear density
from the cable length stored in the selected references. It also requires the
gravity vector in the rectified camera frame and the measured cable diameter.
The right-hand artifact review shows reference observability, motion and shape
excitation, fixed-length precision, and the fitted EI, damping, rollout RMSE,
and refinement history. These are run diagnostics, not a separate evaluation
pipeline.

For extraction and fitting, the UI runs `run_offline_der.py` with an explicit
CLI subcommand in an isolated child process and streams its log. Therefore it
executes exactly the same offline code as the CLI, remains responsive during
CUDA work, and its Cancel button terminates the actual offline process.

## Why the first optimizer does not contain a particle filter

The phase-one identification recordings require a fully visible, contact-free
cable. Their reconstructed centreline and covariance are direct measurements
of the state used by the DDER rollout loss. Putting a particle filter inside
this optimizer would add sampling noise and allow state-estimation error to be
traded against EI or damping, weakening parameter identifiability without
adding information.

The fitted DDER transition is intended for the online particle filter, where a
PF is valuable for crossings and occlusion. A later latent-state identification
experiment may place a smoother or PF inside an EM-style optimizer for
partially observed recordings, but that is a separate extension and should be
compared against this clean-reference baseline.

## Shared mechanics

Offline fitting and future online deployment import the same implementation in
`cable_twin/der.py`. For ordered nodes `q_i`, segment length
`h = L/(N-1)`, unit segment tangents `t_i`, and circular-cable bending
stiffness `EI`, the twist-free geometric curvature binormal and bending energy
are

```text
kb_i = 2 (t_(i-1) x t_i) / (1 + t_(i-1) . t_i)
E_b  = EI/(2 h) sum_i ||kb_i||^2.
```

For small turning angles this agrees with the earlier second-difference energy,
but it remains the correct discrete-rod geometry under large rotations. Forces
are obtained as `-dE_b/dq` with PyTorch autograd. The transition uses lumped
measured mass, gravity, effective viscous damping, semi-implicit integration,
measured endpoint motion as a clamped boundary, and a differentiable coupled
batched Newton/SHAKE projection of every segment to length `h`. At the fixed
small node count this enforces all adjacent constraints together and reaches
the required metric tolerance in fewer CUDA iterations than local PBD passes.

This is the isotropic, twist-free specialization of a differentiable discrete
elastic rod. It is appropriate for the present circular cable because the RGB-D
centreline cannot observe a material frame or torsion. It must not be described
as a learned torsional model. No residual neural dynamics are added in this
phase, so EI and damping remain directly interpretable. The formulation follows
the geometric DER/DDER structure used by
[DEFORM](https://github.com/roahmlab/DEFORM) while deliberately retaining only
the state observable in this experiment.

## Required experiments

Use recordings in which the complete cable remains visible and does not touch
the cube, table, another cable, or the camera frame. Include several endpoint
motions and free transients; a nearly stationary recording cannot identify
dynamics. The fitter checks both velocity and shape-excitation magnitude and
refuses recordings below the configured excitation threshold.

Measure rather than fit:

- cable length and radius;
- total cable mass, then compute `linear_density_kg_m = mass_kg / length_m`;
- the gravity vector expressed in the rectified left-camera frame.

For a level camera using the current `RIGHT_HANDED_Y_UP` convention, gravity is
approximately `0 -9.80665 0`. If the camera is tilted, transform gravity from a
calibrated camera orientation or IMU measurement; do not use the level-camera
value silently.

## 1. Extract reference trajectories

```powershell
.\.venv\Scripts\python.exe run_offline_der.py extract `
  --svo recordings\free_cable_01.svo2 recordings\free_cable_02.svo2 `
  --output-directory data\dder_references `
  --cable both
```

Each requested cable and SVO produces:

- a compressed `.npz` containing timestamps, source positions, fixed-length
  centreline positions, velocities, node observability, covariance, and route
  diagnostics;
- a `.npz.json` containing the recording, calibration, PIDNet, observation,
  extraction, artifact-hash, and software identity.

Only routes with both metric endpoints, sufficient valid registered depth, and
bounded missing arc are eligible. Route identity is selected over complete
contiguous sequences with a Viterbi continuity objective. Centreline length is
projected to exactly 0.518 m; a candidate that does not meet the configured
projection tolerance is rejected rather than retained as approximate data.

## 2. Identify the free-cable model

```powershell
.\.venv\Scripts\python.exe run_offline_der.py fit `
  --reference data\dder_references\free_cable_01_cable0_dder_reference.npz `
              data\dder_references\free_cable_02_cable0_dder_reference.npz `
  --linear-density-kg-m MEASURED_VALUE `
  --gravity-camera-m-s2 GX GY GZ `
  --output data\models\cable_dder_v1.json
```

The optimizer uses covariance-weighted multistep rollouts. Because only two
parameters are fitted, it first performs a bounded coarse physical search and
then differentiable refinement. The exported model is the best full-dataset
checkpoint, not automatically the last epoch.

The deployment JSON contains physical parameters, solver settings, equation
version, reference identities, excitation level, training history, random
seed, final observed RMSE, and the selected checkpoint. The online tracker will
load it through `cable_twin.der.load_dder_model` and use the same transition
under `torch.no_grad()`; it does not use a separately approximated online
physics model.

Do not combine both cables in one fit unless their material, diameter, length,
and linear density are genuinely the same. Do not interpret the fitted damping
as a uniquely separated material/air-drag coefficient; it is the effective
free-cable damping observable from these trajectories.
