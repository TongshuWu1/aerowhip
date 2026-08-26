# DDER-MPPI cable-strike controller

`run_online.py` runs the impact/whip receding-horizon model-predictive
path-integral controller whose prediction model is the identified distributed
discrete elastic rod.

```powershell
.\.venv\Scripts\python.exe run_online.py
```

The UI is an exact-state simulation testbed. It does not ingest Motive data or
send flight commands.

## Continuous figure-eight tracking baseline

`run_figure8_tracking.py` is a separate matched-physics state-information
experiment:

```powershell
.\.venv\Scripts\python.exe .\run_figure8_tracking.py
```

It asks the free material tip to follow a configurable flat figure-eight until
stopped. The reference is a geometric path, not a time-indexed trajectory: it
has no loop frequency or requested traversal speed. The minimal MPCC-style
objective contains running contour error, geometric arc-length progress, a
normalized free-tip acceleration penalty, weak root-acceleration effort and
command smoothness, plus the existing safety terms. It contains no prescribed
tip speed, swing, backtracking, cable-energy, or root-motion reward. The DDER
physics determines which root motion is useful. The UI defaults to 1,024 MPPI
samples, two refinement iterations, and 11 acceleration knots. Acceleration
knots parameterize the root-control trajectory and are independent of the 11
DDER cable nodes.

When changing the prediction horizon, preserve roughly 0.1 s acceleration-knot
spacing (for example, 1 s/11 knots or 2 s/21 knots). Keeping only 11 knots over
a longer horizon coarsens the action trajectory and usually worsens MPPI. The
GPU cost uses a vectorized arc-length-guided local projection, so predictions
can advance around the complete loop without per-frame GPU dispatch or crossing
branch jumps.
The optional initial swing seed always occupies at most its first one second;
longer horizons pad the same smooth zero-net seed rather than stretching its
timing.
The task retains the authoritative 11-node DDER, accelerated
CUDA rollout, translational point-mass plant, and stochastic MPPI update, but
replaces all impact/contact terms with geometric path following plus weak
control regularization and the existing safety terms. Drone displacement and
return-to-start do not appear in the objective. The drone is free to lower,
rise, or move laterally, and MPPI selects that motion through the DDER-predicted
effect on endpoint contour and geometric progress. The path-objective weights
are tunable and none is a hard motion constraint.

The observation selector compares `full` distributed cable positions and
velocities against `endpoint`, which uses only drone/root state and current
free-tip position/velocity. In endpoint mode the interior is the controller's
previous DDER prediction corrected smoothly from root to tip by
`endpoint_conditioned_state`; simulated truth interior nodes are retained only
for visualization and saved evaluation data. EI and Cb are identical in plant
and controller, and adaptation is disabled.

The live and final summaries report drone displacement from its start alongside
tip-tracking error. This quantity is an evaluation diagnostic only; it is not
part of the optimization objective.

## Supported architecture

```text
fitted one-attachment/free-tip cable model
    -> optional mass-conserving simulation-grid reduction
    -> accelerated controller DDER
    -> CUDA-batched MPPI
    -> execute short prefix in an independent simulated plant
    -> observe realized distributed state
    -> shift and replan
    -> optionally adapt EI,Cb between strikes
```

The public controller does not consume a learned policy. MPPI is the MPC method.

## Model artifact

`model.load_cable_model()` accepts the canonical one-attachment/free-tip model
schema and validates:

- exactly one prescribed root vertex;
- 11 ordered measured material points;
- explicit marker-to-DDER-node mapping;
- material coordinates, rest lengths, and masses;
- homogeneous `EI` and `Cb`;
- no fitted torsional parameter in the free-tip centerline artifact.

For pre-data testing it also accepts the latest v5 two-holder fit through an
explicitly labelled provisional conversion. That conversion retains `EI,Cb`,
removes the old terminal-frame constraints and `GJ`, and adds the measured
free-tip marker mass. It is not a substitute for the final one-attachment fit.

## Controller and plant models

`build_controller_and_truth_models()` constructs two immutable snapshots:

- the controller estimate used by MPPI; and
- the hidden simulation truth used by the plant.

They share grid, masses, geometry, timestep, constraints, and substeps. In a
controlled mismatch experiment only `EI` and `Cb` differ. Unit truth ratios
return the exact same model object/hash.

When a lower simulation-node count is selected, masses are deposited onto a
uniform material grid, total mass is conserved, and state is transferred by
material-coordinate interpolation followed by position/velocity constraint
projection. The physical 11-marker provenance is preserved separately; a
reduced node is not presented as a newly measured marker.

## Drone/root model

The drone is currently a translational point mass with bounded acceleration:

\[
p_{k+1}=p_k+\Delta t\,v_k+\tfrac12\Delta t^2 a_k,
\qquad
v_{k+1}=v_k+\Delta t\,a_k.
\]

Its position prescribes the cable root. A faster low-level tracking loop is
assumed. Cable reaction force does not change the point-mass motion. The UI
therefore calls the velocity value a **Safety speed limit**, not an actuator
command limit.

## Task contract

`problem.MpcProblem` contains the target center, desired unit impact direction,
minimum directed tip speed, target sphere radius, impact-cone half-angle, and
safety geometry. The UI currently initializes the minimum speed field to
`3.5 m/s`; it is a saved task parameter, not a universal constant.

A valid strike requires:

1. first swept free-tip entry into the target sphere;
2. no strictly earlier non-tip cable contact;
3. directed tip speed at least the requested value;
4. tip velocity inside the requested direction cone; and
5. no safety violation.

### Continuous contact evaluation

Tip entry and closest approach are solved analytically on every piecewise-linear
physics interval. Impact time, position, velocity, and drone displacement are
interpolated at the event. `impact_frame` remains the upper bracketing frame for
artifact compatibility.

Non-tip geometry uses the full cable segments rather than only DDER vertices.
Moving-segment contact uses 32-step conservative advancement with a one
micrometre tolerance. It prevents temporal tunnelling under piecewise-linear
endpoint motion, but it is a numerical conservative CCD for a bilinear moving
segment, not an analytic surface root. The evaluator detects contact ordering;
it does not apply a physical collision impulse.

## Fixed MPPI objective

The supported objective is defined once as `PUBLIC_MPPI_OBJECTIVE` in
`mppi.py`. `MppiSettings` defaults and the UI both consume it. Historical tools
must pass any different objective explicitly.

The objective contains:

- bounded target-position cost;
- proximity-gated directed-speed deficit;
- proximity-gated impact-direction deficit;
- a large negative valid-strike cost;
- soft drone displacement at the selected event;
- safety penalties for target keepout, speed, ground/altitude, cable-drone
  clearance, actuator bounds, and non-tip-before-tip contact;
- weak control effort and smoothness.

It deliberately contains no explicit cable energy, curvature, shape,
straightness, wind-up, release, reversal, or hard start-centered excursion
term. The UI shows the fixed objective/safety summary read-only.

## MPPI update

The control trajectory has a configurable number of three-dimensional
acceleration knots. Knots are interpolated at the configured command rate. Each
MPPI iteration samples bounded Gaussian perturbations, optionally uses
antithetic pairs, evaluates all candidates with the real objective, and forms
the usual temperature-weighted update. The best sampled trajectory is retained
separately from the weighted nominal trajectory.

Gradient guidance is disabled in the public objective. DDER-gradient studies
remain reproducibility tools and do not run on the online critical path.

## Figure-8 partial-observation experiment

`../run_figure8_tracking.py` is the separate state-information experiment. Its
third observation mode places a causal DDER moving-history observer in front of
the unchanged MPPI controller. The observer accepts attachment position and
velocity, free-tip position, timestamps, and its own recursive prior; it does
not accept the simulator interior state or future measurements. A 0.30-second
endpoint history drives a 24-dimensional smooth state correction through
batched finite-difference GN/LM shooting. `EI` and `Cb` remain fixed and matched
in this experiment. See
`../reports/FIGURE8_DDER_HISTORY_OBSERVER_REPORT.md` for the controlled clean and
hidden-interior-velocity comparison.

The production history replay is now a fixed-shape, full-window CUDA graph.
Full candidate states stay on the GPU, immutable DDER constants are cached,
and redundant nominal/final physical evaluations have been removed.  The
unchanged 16-frame, 24-variable, two-iteration observer measures 24.94 ms mean
and 26.70 ms p95 on the RTX 4080 after prewarm, compared with 248.5 ms in the
frozen prototype.  Numerical equivalence and the implementation breakdown are
documented in `../reports/HISTORY_OBSERVER_ACCELERATION_REPORT.md`.

## Receding-horizon execution

`receding_mppi.run_receding_horizon_mppi()` performs:

```text
current drone + cable state
    -> shifted previous knot sequence
    -> MPPI refinement
    -> execute first interval/block
    -> propagate plant
    -> append realized state
    -> repeat until strike/failure/timeout
```

The first solve may load a verified knot archive. Later solves shift the previous
solution. Every update records its prediction, selected controls, realized
prefix, wall time, rollout count, and terminal diagnostics. The UI renders each
new realized block as it arrives and provides real-time 1x replay after the run.

## CUDA acceleration

The online UI calls `require_online_acceleration()` for both controller and
plant. A supported run requires:

- full-horizon CUDA graph capture;
- fused CUDA event/cost evaluation;
- no silent reference-rollout fallback.

Arbitrary accepted node counts use captured CUDA DDER propagation and fused
cost. Exactly 11 nodes also use the specialized fixed-topology fused mechanics
operator. The UI labels these tiers separately. The reference PyTorch DDER and
CPU objective remain for tests and scientific equivalence checks.

## Between-strike physical adaptation

`online_adaptation.BetweenStrikeAdaptationSession` owns a persistent monitor,
fitter, atomic runtime store, and result history. It does not reconstruct this
state for every strike.

After each strike it:

1. converts the realized exact-state simulation into distributed observations;
2. preserves a monotonic session clock while preventing a segment from crossing
   recording boundaries;
3. updates health, excitation, and information diagnostics;
4. selects short fit and held-out-validation segments;
5. estimates `EI,Cb` in log coordinates using batched forward DDER finite
   differences and a small LM/Gauss-Newton solve;
6. accepts only a held-out improvement;
7. builds and prewarms a new accelerated runtime; and
8. atomically publishes it for the next strike.

The runtime is immutable during an MPPI solve. Fitting never blocks the 10 Hz
planning path. Non-tip-first executions are conservatively excluded from
fitting when an exact contact timestamp is unavailable.

The UI plots absolute `EI` and `Cb` percentage error and joint log error against
hidden simulated truth. It also reports before/after held-out all-node position
RMSE. These plots are evaluator diagnostics, not quantities available in a real
experiment.

## UI structure

The public Tkinter UI separates:

- **Task** — goal and strike definition;
- **Controller** — prediction discretization, MPPI compute, execution rate,
  limits, and first-solve warm start;
- **Plant truth** — simulation-only hidden `EI,Cb` mismatch;
- **Adaptation** — enable/reset, model generation, parameter error, validation;
- **Execution result** — live/replay trajectory and outcome metrics.

Model browsing loads the selected artifact; manually changing its path
invalidates the loaded snapshot. Configuration widgets are locked while a run
is active. Saved output uses the immutable configuration captured at Run time.

Settings profiles use schema `receding_horizon_dder_mppi_settings_v2`. The v1
loader remains compatible and defaults missing adaptation to disabled so an old
fixed-model experiment cannot silently become adaptive.

## Active and legacy boundaries

The supported launcher imports `MpcProblem` from `problem.py`, so CasADi is not
required. `mpc.py`, `realtime.py`, and their GUIs retain earlier IPOPT/CEM
experiments for reproducibility. `trajectory_canvas.py` is the active reusable
viewer; `trajectory_gui.py` retains the earlier standalone experiment. See
`../docs/CODEBASE_MAP.md` for the complete classification.

## Tests

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The suite checks task validation, DDER reduction/transfer, continuous contact,
CPU/CUDA cost agreement, MPPI plan semantics, receding execution, accelerated
runtime requirements, adaptation fitting/publication, profile compatibility,
and UI/catalog contracts.

## Current limitations

- exact simulated distributed state, not live OptiTrack estimates;
- acceleration-tracked point-mass root, not coupled 6-DoF flight;
- no physical target impulse/contact response;
- final one-attachment artifact may not yet be available;
- adaptation has not yet been validated on real marker noise/dropout/latency;
- arbitrary DDER node count is a numerical option, not a change in the physical
  marker layout.
