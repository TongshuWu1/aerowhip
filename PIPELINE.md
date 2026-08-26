# Cable identification and DDER-MPPI pipeline

## Scope

The supported pipeline contains two applications:

```powershell
.\.venv\Scripts\python.exe run_offline_fitting.py
.\.venv\Scripts\python.exe run_online.py
```

The first identifies a homogeneous cable model. The second uses that model in
receding-horizon model-predictive control and can adapt `EI,Cb` between simulated
strikes. No policy-learning stage is part of this pipeline.

## 1. Canonical OptiTrack experiment

### Geometry and observations

The cable has one rigid-body attachment followed by ten ordered markers
`c1...c10`. The root pivot is material point zero and `c10` is the free tip.
The CSV parser preserves marker identity and validity; new unlabeled Motive
points are not substituted for missing labeled markers.

The fitted DDER may contain more simulation vertices than measured material
points. The artifact stores both material coordinates and the explicit
marker-to-node observation map. Changing simulation resolution never creates
new physical observations.

### Boundary condition

Only the root position is prescribed. The root tangent and the distal endpoint
remain dynamic. The free material-frame boundary eliminates quasistatic twist
from the circular/isotropic centerline model, so the fitted parameters are:

\[
\theta=(EI,C_b).
\]

### Data selection and fitting

Every Training take is treated by the same objective; take categories are not
used to fit separate parameters. Candidate windows must pass marker validity,
continuity, chord, speed, and numerical checks. The UI audit can explicitly
move unusable takes to `Unused`. It does not require each take to contribute a
minimum number of windows.

Each fitting rollout is 100 frames. This is approximately one second for the
intended 100 Hz Motive data. Multiple shooting initializes each rollout from
the measured distributed state and prescribes the measured root motion. The
loss uses all observed material points; Validation takes never influence the
selected parameters.

The result is a versioned JSON artifact containing geometry, masses, material
coordinates, solver settings, parameters, data/config provenance, fit history,
and held-out metrics.

## 2. Online simulation model

### State and boundary input

The controller state contains drone position/velocity and all DDER vertex
positions/velocities. The drone command is a bounded three-dimensional
acceleration. A faster low-level tracker is assumed, so the root follows a
double-integrator update and prescribes the cable attachment position. Cable
reaction is not fed back into the drone point-mass plant.

The DDER uses the identified mass, length, diameter, rest lengths, gravity,
`EI`, and `Cb`, with the same inextensibility projection and fixed numerical
settings stored in the model/runtime. Eleven observed material points remain
fixed. Refinement factor 1, 2, or 3 subdivides each measured interval into the
same number of homogeneous DDER segments, producing 11, 21, or 31 simulation
nodes. Distributed bare-cable mass is re-lumped on the chosen grid while
discrete marker masses stay at their measured material coordinates.

### Acceleration paths

The supported online controller requires CUDA full-horizon graph capture and a
fused CUDA cost evaluator. All three public meshes use topology-specialized
damping and four-plus-one projection kernels. The application reports:

```text
maximum (captured horizon + fused 11-node mechanics)
maximum (captured horizon + fused 21-node mechanics)
maximum (captured horizon + fused 31-node mechanics)
```

It raises an error instead of silently using the reference PyTorch rollout.

## 3. Receding-horizon MPPI

Let `U` be the sequence of 3-D acceleration knots. At update `t`, the previous
solution is shifted to create the nominal plan. MPPI samples perturbations
`epsilon_i`, simulates every candidate through DDER, evaluates the real task
cost `J_i`, and applies the path-integral weighted update. It then executes only
the first control interval/block and replans from the realized distributed
state.

The initial solve may use a warm start. Its JSON sidecar is checked against the
loaded source-model hash, target, impact contract, horizon, command rate, and
acceleration limit; a mismatch remains usable as an explicitly warned
exploratory seed. Subsequent solves do not start from zero; they shift the
previous solution. Warm starts are an initialization,
not a prescribed motion phase.

### Continuous target event

For every candidate, the free-tip path is piecewise linear between physics
frames. The evaluator finds the first analytic point/sphere entry. If no entry
occurs, it uses the continuous closest point on all intervals. Impact position,
velocity, drone displacement, and time are interpolated at that event.

Non-tip contact is evaluated on the complete cable segments, not only the
vertices. Between frames, conservative advancement detects a moving bilinear
segment crossing the target sphere. A strictly earlier non-tip contact
invalidates the strike; simultaneous entry of the final segment and tip is
allowed. This is continuous geometric event evaluation, not rigid-body impact
response.

### Versioned, configurable objective

With event distance `d`, directed tip speed `v_parallel`, direction cosine `c`,
and a proximity gate `g_p(d)`, the public objective contains:

\[
J = J_{pos}+J_{speed}+J_{dir}+J_{success}+J_{disp}+J_{safety}+J_u.
\]

The bounded position term is:

\[
J_{pos}=w_p\frac{d^2}{d^2+\sigma_p^2}.
\]

Speed and direction deficits are gated near the target:

\[
J_{speed}=w_v g_p(d)[v^*-v_{parallel}]_+^2,
\]

\[
J_{dir}=w_\theta g_p(d)[\cos\theta_{max}-c]_+^2.
\]

A valid tip-first strike receives a large negative success cost. Soft drone
displacement, physical safety penalties, control effort, and control
smoothness complete the cost. There is no cable-energy, curvature, shape,
wind-up, release, reversal, or hard start-centered excursion term.

`PUBLIC_MPPI_OBJECTIVE` in `drone_mpc/mppi.py` defines the public model-level
defaults. The online UI exposes the strike weights, distance/gating scales,
weak predictive-speed shaping, safety weight, displacement weight, and control
regularization in a dedicated **Objective** tab. Pressing **Run** creates an
immutable `MppiSettings` snapshot, so a running solve cannot silently change
objective; settings profiles and execution artifacts store the complete values.
The validated UI baseline keeps predictive-speed shaping disabled, matching
what the earlier online GUI actually executed.

## 4. Between-strike EI/Cb adaptation

Adaptation is outside the MPC critical path:

```text
control with frozen model
    -> record distributed motion
    -> persistent health/excitation/information monitoring
    -> select short informative fit and validation segments
    -> estimate EI,Cb
    -> held-out validation
    -> rebuild and prewarm accelerated runtime
    -> atomic publication for the next strike
```

The monitor and informative cache persist across strikes. Local recording clocks
are converted to a monotonic session clock, while recent-history boundaries are
cleared so a fit segment cannot span unrelated initial states. Contact-affected
and unsafe data are excluded; when a precise non-tip event timestamp is absent,
a non-tip-first execution is conservatively excluded in full.

An intentional plant-truth change is a new physical regime, not a new adaptation
session. The published EI/Cb estimate, generation history, strike count, and
parameter-error graph persist. Recent motion, health hysteresis, and the
informative cache are rearmed/cleared so identification never mixes segments
recorded under two different truth plants. The plot marks the regime boundary.

Only `EI,Cb` are adapted. The fit uses short distributed-state segments,
batched central finite differences in log-parameter coordinates, a 2-by-2
Levenberg-Marquardt/Gauss-Newton update, trust limits, batched line search, and
held-out acceptance. A strike sees one immutable runtime generation.

The current online UI feeds exact simulated distributed state to this adapter.
The causal OptiTrack observation/state-estimation path is a separate validation
stage and is not represented as complete physical integration.

## 5. UI and provenance

The online UI separates:

- **Task:** initial position, target, impact vector/speed/radius/cone;
- **Controller:** horizon, rates, node count, knots, samples, iterations,
  noise, seed, acceleration and safety-speed limits, full/endpoint feedback,
  warm start;
- **Objective:** strike, gating, safety, and control-regularization weights;
- **Plant truth:** hidden simulation-only `EI,Cb` ratios;
- **Adaptation:** enable/reset, published generation, parameter error plot,
  and held-out prediction change.

All active-strike configuration is immutable. Most widgets are locked during a
run; the two plant-truth fields remain editable only to queue the next strike's
regime. Editing the visible model path invalidates the loaded snapshot. Saved
executions use immutable run provenance, not fields edited after completion.
Settings profiles are versioned; legacy profiles without adaptation retain
fixed-model semantics.

## 6. Outputs

Offline output:

```text
optitrack_offline/models/cable_model.json
```

Online profiles and executions default to:

```text
data/drone_mpc/settings_profiles/
data/drone_mpc/receding_mppi/
```

The latter directories are working data and are intentionally not source-code
artifacts. A reproducible study must record the model hash, normalized settings,
resolved random seed, warm-start provenance, controller/truth model hashes,
acceleration tier, and runtime generation.

## 7. What this pipeline does not claim

- no live Motive/OptiTrack ingestion in the public online UI;
- no measured cable-velocity estimator in the public online UI;
- no force/torque-coupled flight dynamics in the MPPI plant;
- no physical collision impulse/contact response;
- no online fitting inside the 10 Hz planning deadline;
- no adaptation beyond `EI,Cb`;
- no proof that the provisional two-holder artifact is the final free-tip model;
- no learned control policy.

Earlier RGB/ZED particle filtering, IPOPT/CEM controllers, solve-once tools, and
other ablations remain research/legacy utilities. They are not public execution
paths. See [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md).

## 8. Regression contract

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The regression suite covers artifact validation, DDER propagation, reduction
and state transfer, MPPI event/cost semantics, continuous contact, receding
execution, CUDA/reference agreement, runtime publication, distributed
adaptation, offline parsing/fitting, and UI/catalog contracts.
