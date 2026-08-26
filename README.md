# Cable Twin

Research software for identifying a flexible cable from OptiTrack data and
using the identified distributed discrete elastic-rod model inside
receding-horizon MPPI control.

The supported workflow is deliberately narrow:

```text
OptiTrack takes
    -> offline one-attachment/free-tip EI,Cb identification
    -> accelerated full-state DDER-MPPI simulation
    -> optional EI,Cb adaptation between strikes
```

MPPI is the project's model-predictive controller. There is no learned-policy
stage in the supported codebase.

## Public applications

There are two public desktop applications:

```powershell
.\.venv\Scripts\python.exe run_offline_fitting.py
.\.venv\Scripts\python.exe run_online.py
```

`run_offline_fitting.py` reviews and fits OptiTrack recordings.
`run_online.py` runs the accelerated simulation controller and the optional
between-strike physical-parameter adapter. Both UIs use a plain, research-first
Tkinter layout. The small research console is an optional launcher/status view,
not a third scientific workflow.

The separate first-stage state-information experiment is launched with:

```powershell
.\.venv\Scripts\python.exe .\run_figure8_tracking.py
```

It follows a configurable flat figure-eight geometric path with the free cable
tip, without prescribing loop frequency or speed. The compact UI compares
matched-physics full distributed-state MPPI
against the existing instantaneous endpoint-only state conditioner and a
causal moving-history DDER observer. The observer receives only attachment
motion and free-tip position history, reconstructs a full distributed state,
and passes that estimate to unchanged MPPI. It uses the same 11-node
accelerated DDER and translational acceleration-control plant, and does not run
impact logic or physical-parameter adaptation. Runs are logged to
`data/drone_mpc/figure8_tracking/` as CSV, full-truth NPZ, and JSON metadata.
The controlled hidden-state result is documented in
[`reports/FIGURE8_DDER_HISTORY_OBSERVER_REPORT.md`](reports/FIGURE8_DDER_HISTORY_OBSERVER_REPORT.md).

## Offline OptiTrack identification

The canonical experiment has one rigid-body cable attachment and ten ordered
material markers `c1...c10`; `c10` is the free tip. Motive exports positions at
approximately 100 Hz. The attachment position is prescribed. Its orientation,
the attachment tangent, all flexible-node positions, and the distal endpoint
are prediction targets rather than additional boundary constraints.

The fitter estimates one homogeneous bending stiffness `EI` and one
Kelvin-Voigt curvature-rate damping parameter `Cb` from all Training takes.
Validation takes remain held out. Fitting uses fixed 100-frame rollouts, which
are approximately one second only for 100 Hz recordings. The UI exposes take
roles, a clearly mutating **Audit and exclude unusable takes** action, model
geometry/mass, optimizer settings, fit progress, and validation playback.

For the straight circular/isotropic free-tip baseline, quasistatic material
twist is eliminated by the free material-frame boundary condition. This does
not mean the physical cable has zero torsional stiffness; it means `GJ` is not
identifiable or needed in this reduced centerline experiment.

The fitted artifact is:

```text
optitrack_offline/models/cable_model.json
```

See [optitrack_offline/README.md](optitrack_offline/README.md) for the capture,
CSV, filtering, fitting, and held-out validation contracts.

## Online DDER-MPPI

Launch the controller with:

```powershell
.\.venv\Scripts\python.exe run_online.py
```

At each MPC update the controller:

1. starts from the current drone state and complete distributed cable state;
2. shifts the previous acceleration-knot plan;
3. samples bounded three-dimensional knot perturbations;
4. propagates every candidate through the full DDER cable model on CUDA;
5. evaluates the fixed impact and safety objective;
6. performs the MPPI importance-weighted update;
7. executes only the next short control prefix; and
8. replans from the newly realized cable state.

The free-tip target event is evaluated continuously between physics frames.
Tip entry into the target sphere and continuous closest approach are analytic
for the piecewise-linear tip path. Non-tip target contact checks the complete
cable segments and uses conservative continuous collision detection between
frames. A valid strike requires tip-first entry, the requested directed speed,
the requested direction cone, and no safety violation.

The objective contains target position, directed speed, impact direction, a
large valid-strike bonus, soft drone displacement, safety penalties, and weak
control effort/smoothness. It contains no cable-energy reward, prescribed
shape, wind-up phase, release phase, reversal reward, or hard maximum drone
excursion. The UI displays the fixed objective and safety contract read-only;
task and compute controls remain separate.

### Accelerated runtime

The public online path requires:

- CUDA full-horizon DDER graph capture;
- the fused CUDA MPPI event/cost evaluator; and
- batched candidate propagation.

Those paths work for every node count accepted by the loaded artifact. The
public refinement factors 1/2/3 additionally use topology-specialized fused
11/21/31-node DDER mechanics. The UI reports which acceleration tier is active
and refuses to fall back silently to the slow reference path.

### Between-strike adaptation

The simulation UI can use hidden plant ratios for `EI` and `Cb`. The controller
does not receive those ratios. During a strike the controller model is frozen.
After the strike, a persistent monitor retains recent history, informative
segments, hysteresis, and cooldown state. If mismatch and information gates
justify a fit, short distributed-state segments are fitted and held-out
validated. An accepted model is rebuilt, prewarmed, and atomically published
for the next strike.

The Adaptation tab plots the published `EI` and `Cb` error against simulated
truth after every strike and reports held-out all-node prediction RMSE. That
truth comparison is a simulation diagnostic only; a physical experiment will
not know the true parameters.

Settings profiles are stored under:

```text
data/drone_mpc/settings_profiles/
```

Profiles are versioned. Older profiles that do not contain an adaptation field
retain their original fixed-model behavior; adaptation is not silently enabled.

## Current scientific boundary

The online application is currently a simulation testbed, not a flight stack:

- cable feedback is exact distributed simulator state;
- hidden truth mismatch changes only `EI` and `Cb`;
- the drone is an acceleration-tracked point mass;
- cable reaction force does not alter the drone dynamics;
- no live Motive stream, state estimator, radio, or flight command is used;
- the tracked model may still be the explicitly labelled provisional transfer
  from the older two-holder fit until the final one-attachment dataset is fit.

The sensing-aware synthetic study and actual OptiTrack integration remain
separate validation stages. Arbitrary simulation-node counts must not be
mistaken for arbitrary physical marker counts: the physical observation
contract remains the ordered 11-material-point experiment.

## Isaac plant validation

The separate Isaac Lab runner is a plant-implementation validation tool, not a
public workflow UI:

```powershell
.\run_isaac_whip.bat --num-envs 1 --mode hover
.\run_isaac_whip.bat --num-envs 64 --mode excite
```

It provides a 6-DoF force/torque-driven drone and passive cable for simulator
checks. See [isaac_whip/README.md](isaac_whip/README.md) and
[ISAACSIM_SETUP_INSTRUCTIONS.md](ISAACSIM_SETUP_INSTRUCTIONS.md).

## Verification

Run the complete CPU regression suite with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

CUDA-specific acceleration/contact tests skip when no compatible CUDA device is
available. Supported launchers can also be imported without CasADi; CasADi/IPOPT
is retained only by explicitly legacy research utilities.

## Code organization

```text
optitrack_offline/  CSV parsing, take review, EI/Cb fitting, validation UI
drone_mpc/          DDER plant, MPPI, receding controller, adaptation, online UI
cable_twin/         shared DDER/perception code and retained earlier prototypes
research_tools/     reproducibility, profiling, and ablation scripts
research_ui/        optional two-stage launcher/status console
isaac_whip/         separate Isaac Lab plant validation
reports/            generated research and engineering reports
docs/               current method handoffs and codebase map
tests/              regression and numerical-contract tests
```

The authoritative active/research/legacy module map is in
[docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md). The detailed method is in
[PIPELINE.md](PIPELINE.md).
