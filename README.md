# Cable Twin

One shared differentiable constrained elastic-rod model with separate offline
and online workflows. The canonical OptiTrack experiment prescribes one
attachment position and predicts a mechanically free distal end.

## Public applications

The project has three supported workflow applications:

```powershell
.\.venv\Scripts\python.exe run_offline_fitting.py
.\.venv\Scripts\python.exe run_sac_training.py
.\.venv\Scripts\python.exe run_online.py
```

They form one explicit experimental sequence:

1. **Physical model identification** reviews OptiTrack takes and publishes the
   canonical cable-model artifact.
2. **Goal-conditioned policy learning** trains SAC under that frozen nominal
   model and retains validated-best and latest checkpoints.
3. **Online controller** runs full-state receding-horizon DDER-MPPI with the
   frozen cable model and a matched, independently constructed simulation plant.

The third application is the nominal matched-model control baseline. It is not
yet a live OptiTrack interface, online parameter adaptation experiment, or a
force-coupled flight result.

The separate Isaac Lab plant runner is an implementation-validation tool, not
a fourth workflow UI. It provides a 6-DoF force/torque-driven drone with the
identified passive cable in one or 64 cloned environments:

```powershell
.\run_isaac_whip.bat --num-envs 1 --mode hover
.\run_isaac_whip.bat --num-envs 64 --mode excite
```

Its physical assumptions, installation contract, and validation limitations
are documented in [isaac_whip/README.md](isaac_whip/README.md). To reproduce
the complete RTX 5090 environment on another computer, give that computer's
Codex agent [ISAACSIM_SETUP_INSTRUCTIONS.md](ISAACSIM_SETUP_INSTRUCTIONS.md)
and ask it to execute the file from beginning to end.

## Legacy planar identification prototype

This earlier image-plane workflow is retained as an internal prototype under
`research_tools.legacy_rgb_offline`; it is not a public project entry point.

The offline workflow uses 2D PIDNet curves, not depth or a point cloud. Keep the
camera fixed, move the fully visible cable approximately parallel to the image
plane, and record one or more endpoint-driven motions. One constant metric scale
per recording is obtained from the known cable length; the resulting trajectories
identify the shared bending parameters `EI` and `Cb`.

One model is published per cable:

```text
data/offline_dder/models/cable1_dder.json
```

See [PIPELINE.md](PIPELINE.md) and
[cable_twin/offline/CABLE_MODEL.md](cable_twin/offline/CABLE_MODEL.md).

The compact internal parameter-recovery check is:

```powershell
.\.venv\Scripts\python.exe -m research_tools.model_check
```

It saves `data/offline_dder/synthetic_recovery.json`. This uses known synthetic
parameters to check the implementation; it is not experimental validation.

## OptiTrack offline identification

Review labeled Motive CSV takes and jointly fit cable material parameters with:

```powershell
.\.venv\Scripts\python.exe run_offline_fitting.py
```

The application assigns dynamic takes to Training or held-out Validation and
fits one homogeneous bending stiffness `EI` and curvature-rate damping `Cb`.
Each new 100 Hz Motive CSV contains one rigid body whose pivot is at the cable
attachment and ten ordered markers `c1...c10`; `c10` is on the free tip. The
measured pivot position is prescribed, while its orientation, the attachment
tangent, and every flexible-node position remain unprescribed. The free-tip
trajectory is a prediction target, not a second boundary.

For the naturally straight circular/isotropic baseline, quasistatic torsion is
analytically eliminated by the free material-frame boundary condition. This is
not an assumption that the physical cable has `GJ = 0`; `GJ` is absent from
the reduced centerline problem. One-second rollouts are fitted with bounded
global initialization followed by projected CUDA-batched Adam, and
full-Training evaluations select the artifact. Validation reports
attachment-only open-loop prediction and periodic full-cable position
observations. See
[optitrack_offline/README.md](optitrack_offline/README.md) for the definitive
capture and validation contract.

The earlier two-holder `c1...c9` takes and twist-aware
`EI`/`GJ`/`Cb` artifact are incompatible with the new boundary condition.
They cannot initialize, train, or validate the free-tip model; new
one-pivot/free-tip captures require a fresh fit.

## Drone whip simulation and MPC

The matched-model controller experiment is launched with:

```powershell
.\.venv\Scripts\python.exe run_online.py
```

It uses the same full fitted DDER for CUDA-batched MPPI planning and a freshly
constructed independent plant. MPPI searches smooth 3-D acceleration knots,
executes only a short prefix, observes the current distributed cable state,
shifts the previous plan, and replans.
The event cost is defined only by first tip contact/closest approach, requested
directed impact speed and cone, soft drone displacement and safety penalties,
and weak control regularization. It has no reward for whipping, cable energy,
shape, straightness, or prescribed wind-up/release phases; the maneuver must
emerge from the cable physics. The UI separates task definition from MPPI
compute settings and shows the fixed full-impact objective read-only. The old
solve-once and IPOPT interfaces remain research utilities rather than online
controls. See
[drone_mpc/README.md](drone_mpc/README.md) for its assumptions and controls.
Until the free-tip fit is available, the UI can use the latest two-holder
`EI`/`Cb` result as an explicitly labelled provisional transfer.

Run the fixed-seed full-plant controller check with:

```powershell
.\.venv\Scripts\python.exe -m research_tools.controller_benchmark
```

It saves compact JSON studies of 7/11/15/21-node controller resolution and
controlled `±30%` `EI`/`Cb` mismatch under identical task, seed, and state
feedback. A separate recorded planning margin tightens the internal predicted
hit tolerance; executed success retains the original physical tolerance.

The fixed-target control milestone is one offline-optimized, physically
verified whip. Run:

```powershell
.\.venv\Scripts\python.exe -m research_tools.whip_optimizer
```

The optimizer searches a smooth forward/recoil motion using a mass-conserving
7-node rod, then refines and verifies it on the exact 21-node fitted rod. Only
the exact-model result is reported. This single-trajectory milestone established
the simulator/controller contract used by the multi-goal demonstration and SAC
stages below.

Train the goal-conditioned closed-loop policy with SAC and future-goal HER:

```powershell
.\.venv\Scripts\python.exe run_sac_training.py
```

The active trainer stores both real transitions and coherently relabeled
episode prefixes in one uniform replay buffer. Future achieved cable-tip
position and velocity define the hindsight goals. Rewards, termination, and
target-dependent safety are recomputed after relabeling. It does not load MPC
demonstrations, pretrain the actor, switch scripted motion phases, or use a
curriculum. Earlier MPC demonstration artifacts remain historical experiments,
not inputs to this baseline.

The nominal actor receives the complete simulated DER state, drone state,
target, required impact direction and speed, remaining time, and previous
command in a target-aligned frame. It outputs one bounded 3-D acceleration
every 20 ms. Training uses CUDA-vectorized DDER rollouts at 100 Hz on a 15-node grid
selected by a matched 7/11/15/21-node reachability audit. Twin-critic SAC
optimizes the four-second task. A valid impact can occur at any physics frame.
The dominant learning event is first contact with the physical target at the
requested velocity; a bounded tip-to-target potential, small time/effort costs,
and safety costs provide secondary shaping.

Contact is detected by sweeping the cable tip between physics frames against a
metric target sphere. Success requires that first contact to satisfy the
requested directed speed and velocity cone. Cable extension is intentionally
not a separate success condition: target position and impact velocity are the
task definition.

The ground-up UI baseline initially fixes the target at 0.80 m horizontal
distance and 0.10 m below the initial drone, with a radial horizontal impact
vector. This isolates whether the learner can discover a whip before target
ranges are broadened. The worker still accepts distance, height, azimuth, and
impact-vector ranges for the subsequent multi-goal experiment. State and
actions use a target-aligned frame while the cable and control remain fully
three-dimensional.

There is no start-centered excursion limit. Successful impacts receive a
secondary squared penalty on drone displacement from the episode start and the
drone must respect target keepout, but neither is a hard bound on maximum path
excursion. Therefore "beyond initial cable reach" is a geometric reporting
category, not by itself proof of whipping or of reach beyond a bounded drone
workspace. That stronger paper claim requires a separate evaluation with a
hard workspace constraint.

The learning algorithm uses a squashed stochastic actor, two critics and target
critics, automatic entropy-temperature tuning, running observation statistics,
and uniform batches from the combined real/HER replay. There is no behavior
cloning, demonstration replay, guided action prior, or curriculum.

Periodic deterministic validation uses fixed seed `seed + 1000` to choose the
deployable checkpoint. Its primary rank is the arithmetic mean of success in
the within- and beyond-initial-reach strata; aggregate success, lower position
error, higher directed speed, and lower impact drone displacement are ordered
tie-breakers. After training, that retained checkpoint is tested with the
separate fixed seed `seed + 10000`; both seeds and all metrics are stored in the
JSON log.
For a multi-training-seed experiment, `--checkpoint-validation-seed` and
`--final-test-seed` freeze those target sets independently of network and
replay randomness.

For interactive training, use the dedicated dashboard:

```powershell
.\.venv\Scripts\python.exe run_sac_training.py
```

It runs SAC in a separate CUDA process and plots deterministic success, tip
error, directed impact speed, unsafe-episode rate, and drone displacement at
impact. It keeps validated-best and latest-policy artifacts distinct. The
optional endless mode trains until `Stop safely` is pressed; stopping saves the
latest policy and incremental JSON history without overwriting the best policy.
There is deliberately no Resume button because the saved policy does not
contain replay, optimizer, environment, or RNG state.

For historical context, an earlier prior-replay experiment held the model,
task, SAC settings, validation seed, and 256-goal final test fixed while
expanding the prior from four to twelve demonstrations:

| Verified prior | Prior transitions | Selected transition | Overall success | Within reach | Beyond reach | Mean minimum error | Directed speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 demonstrations | 28 | 340,096 | 36.3% | 55.3% | 4.2% | 244.6 mm | 2.00 m/s |
| 12 demonstrations | 79 | 180,096 | 59.8% | 55.9% | 66.3% | 154.4 mm | 3.80 m/s |

The twelve-demo policy also increased mean drone displacement at impact from
0.448 m to 0.673 m. These are nominal simulation results, not a flight
controller or proof of whipping under a bounded vehicle workspace. This is a
paired single-seed comparison of the complete demonstration-set intervention:
the overlapping demonstrations were re-optimized as well as eight goals being
added, so it does not isolate demonstration count alone.

Repeating the twelve-demo run with training seeds 42, 43, and 44 while fixing
checkpoint targets to seed 1042 and all 256 final targets to seed 10042 gave:

| Training seed | Overall | Within reach | Beyond reach |
|---:|---:|---:|---:|
| 42 | 59.8% | 55.9% | 66.3% |
| 43 | 59.0% | 59.0% | 58.9% |
| 44 | 48.0% | 34.8% | 70.5% |
| Mean +/- sample SD | **55.6 +/- 6.6%** | **49.9 +/- 13.2%** | **65.3 +/- 5.9%** |

The far-target behavior repeats across all three seeds, but near-target success
and miss magnitude remain seed-sensitive. The internal
`research_tools.sac_multiseed` utility writes the machine-readable aggregate;
it verifies identical models, tasks, demonstrations, settings, and held-out
target strata before combining logs.

For the non-learning predictive-control application, launch:

```powershell
.\.venv\Scripts\python.exe run_online.py
```

The controller uses the fitted DDER model. By default the simulated plant uses
the identical model; the separate `Plant truth` tab can instead scale plant
`EI` and `Cb` while leaving the controller nominal model unchanged. At every
update, CUDA-batched MPPI starts from the current distributed cable state,
shifts the previous solution, optimizes the remaining 3-D acceleration knots,
executes only a short prefix, and replans. The Tkinter UI separates the strike
task from controller compute. It exposes prediction horizon, physics/control/
replanning rates, DDER simulation-node count, samples, MPPI iterations, knot
count, noise, seed, and vehicle limits. Plant-truth ratios are kept outside the
controller tab because they define the hidden simulated experiment, not an MPPI
tuning parameter. Full-state feedback, maximum CUDA batching, and the validated
strike cost are fixed in the public pipeline rather than mixed into routine controller
operation. Ratios `1.0, 1.0` are the exact matched-model baseline. For mismatch
tests, controller and plant retain the same node grid, mass, geometry, timestep,
and solver configuration; only the two physical parameters change, and both
model identities are saved. The legacy spherical drone-excursion constraint is
disabled; drone displacement at impact remains a soft cost. SAC, online
adaptation, DDER gradients, and IPOPT are deliberately excluded from this
controller; the optional truth mismatch is fixed for the run. During execution,
every completed MPC update streams its newly
realized cable frames to the viewport step by step; after completion, the full
maneuver can be replayed in wall-clock-synchronized real time. One CUDA batch provides maximum rollout
parallelism, with presets from 128 low-latency samples to the measured
2,048-sample throughput knee on the development RTX 4080. No live OptiTrack
stream is consumed yet. The drone plant is
acceleration-tracked rather than force-coupled; see
[drone_mpc/README.md](drone_mpc/README.md) for the measurement, provenance, and
physics contracts.

The online UI can save and load a versioned JSON settings profile containing
the model/warm-start inputs, task, controller compute settings, and hidden plant
truth. Profile loading is all-or-nothing: every required field is validated
before any visible setting is changed. Profiles are stored by default in
`data/drone_mpc/settings_profiles/`.

## Legacy ZED particle-filter prototype

After a compatible free-tip artifact is fitted, the online PF can transfer its
homogeneous `EI` and `Cb` to a cable of the same construction. Cable length,
mass, marker masses, and discretization remain experiment-specific and are not
silently reused. The image-only online path uses the same reduced centerline
model and does not consume a legacy `GJ` estimate.

PIDNet supplies the visible 2D route and endpoints, the ZED supplies registered
depth for metric initialization and visible endpoint translations, and the
fitted constrained rod advances every 3D particle. The recurring body
likelihood projects each particle into the image instead of reconstructing a
noisy depth centerline. The ZED viewport highlights PIDNet-selected point-cloud
samples orange and the MAP reconstructed centerline green. See
[cable_twin/online/README.md](cable_twin/online/README.md).

This prototype is retained internally as `research_tools.legacy_zed_online`; it is
not the public `run_online.py` application.

## Layout

```text
cable_twin/
  shared/   PIDNet, route geometry, constrained rod dynamics, model artifact
  offline/  image-plane 2D extraction, EI/Cb fitting, review UI
  online/   live 3D particle filter and replay
optitrack_offline/  one-pivot/free-tip review, EI/Cb fit, held-out validation
drone_mpc/          identified-cable simulation and drone trajectory MPC
```
