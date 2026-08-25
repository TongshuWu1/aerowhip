# Drone whip simulation and MPC

The model-based controller GUI is retained as an internal reproducibility
utility rather than a public project entry point:

```powershell
.\.venv\Scripts\python.exe -m research_tools.mpc_gui
```

Install the bundled IPOPT interface once in the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r drone_mpc\requirements.txt
```

The preferred input is the current `optitrack_one_attached_free_rod_v1`
artifact. Before that dataset is ready, the application also accepts exactly
the latest `optitrack_twist_aware_rod_v5` fit as a visibly marked provisional
model. This transfer retains its learned `EI` and `Cb`, removes `GJ` and both
old terminal-frame constraints, and adds one measured marker mass at the newly
free tip. It is useful for controller development but is not free-tip model
validation and must not be reported as such.

## Model

The fitted artifact remains the source of cable length, mass, `EI`, `Cb`,
gravity, and constraint mechanics. The simulated plant retains this complete
fitted discretization. For control, a separate model is reduced onto a uniform
material grid (seven nodes by default). The full-resolution comparison retains
the fitted grid exactly. Reduction conserves total mass, keeps `EI` and `Cb`
unchanged by default, and records the source-model hash. Each controller uses
the smallest explicit-solver substep count that is stable at the MPC time step.
At each replan the current full state is interpolated in material coordinates
and projected onto the reduced model's exact position and velocity constraints.
The UI exposes explicit controller/plant `EI` and `Cb` ratios for controlled
mismatch tests; these ratios never modify the full plant.

The drone attachment is a fixed vertical offset beneath the drone position.
Only that attachment position is prescribed; the cable tip is dynamic.

The initial cable is the exact vertical hanging configuration. The drone is a
closed-loop translational double integrator:

```math
p_{k+1}=p_k+\Delta t v_k+\tfrac12\Delta t^2 a_k,
\qquad
v_{k+1}=v_k+\Delta t a_k.
```

Here `a` is the net world-frame acceleration assumed to be tracked by a faster
attitude/thrust controller. Acceleration has a hard norm bound; maximum speed
is a path constraint in the MPC. Cable reaction does not yet feed back into
drone acceleration. This approximation must be checked using the measured
cable/drone mass ratio and flight tracking error before experiments.

## Live matched-model receding-horizon MPC

`run_online.py` launches the non-learning baseline. The simulated plant and MPC
use the same fitted DDER model; hidden parameter mismatch and SAC are not part
of this experiment. By default the model advances at 100 Hz, controls update at
50 Hz, the prediction horizon is `N=20` control steps, and `M=5` controls are
applied before replanning. The CUDA worker predicts through those `M` active
commands, solves another fixed-`N` problem, and publishes it atomically. The UI
reports solve time, real-time factor, and missed `M`-step deadlines.

In flight, a full-state OptiTrack estimate replaces the simulated state and the
leading acceleration command is sent to the drone's low-level controller. The
current point-mass drone still assumes that this inner loop tracks translation
acceleration and neglects cable reaction; that approximation requires physical
validation.

Following the low-dimensional action design used in Planar Robot Casting, the
online nonlinear program does not optimize one unrelated acceleration vector at
every knot. IPOPT optimizes a target-aligned forward-recoil primitive: forward
elevation, recoil-plane deflection, forward excursion, recoil excursion,
reversal time, total motion time, and (when enabled) continuous impact time. A
quintic segment drives the drone toward the forward waypoint, and a second
segment reverses it toward a backward waypoint. Both waypoints lie inside the
fixed mission-start excursion sphere. The decoded accelerations are hard
bounded. Hit position, directed speed, impact cone, drone excursion, target
keepout, maximum drone speed, forward stroke, and recoil are explicit IPOPT
inequality constraints evaluated with the deployed DDER rollout.

The DDER equations are not duplicated in a symbolic surrogate. IPOPT receives
central finite-difference Jacobians in the small normalized casting space; all
lower/upper perturbations are evaluated together as one CUDA batch. IPOPT uses
a limited-memory BFGS Hessian approximation. The UI exposes iteration limits,
desired and acceptable tolerances, wall-time limits, and the normalized
finite-difference step.

The UI specifies a mission timeout, metric hit tolerance, minimum directed
world-frame tip speed, and direction-cone half-angle. The event is tested at
every physics frame, including frames between control knots. Forward injection
and recoil are enforced by the two-sweep action structure. Their amplitudes are
optimized, while one absolute phase clock preserves the user-declared switch
across replans instead of restarting injection. The maneuver is locked to the
vertical plane containing the start and target, preventing changing replan
planes from producing a spiral. The development task defaults to a
1.5 m/s minimum directed impact speed. The target is 0.48 m horizontally from
and exactly 0.20 m below the initial drone, so the desired strike is predominantly
horizontal rather than a gravity-driven pendulum swing. The
drone excursion is limited to 0.15 m (15.6% of the 0.961 m provisional cable),
so the vehicle cannot translate to the target; the free tip must be dynamically
extended.

For a stationary target, define

```math
e_k=\lVert p_{tip,k}-p^*\rVert,
\qquad
s_k=d^{*T}v_{tip,k},
\qquad
\theta_k=\cos^{-1}\!\left(
\frac{d^{*T}v_{tip,k}}{\lVert v_{tip,k}\rVert}
\right).
```

An impact event is valid only if

```math
e_k\leq\epsilon_p,
\qquad
s_k\geq v_{min},
\qquad
\theta_k\leq\theta_{max}.
```

For diagnostics, the forward and recoil stroke measures are

```math
D_f(k)=\max_{j\leq k}d_h^T(p_{drone,j}-p_{start}),
\qquad
D_r(k)=D_f(k)-d_h^T(p_{drone,k}-p_{start}).
```

The physical position and direction-cone limits remain unchanged during
evaluation. Planning uses separately recorded robustness margins that tighten
both limits internally. This does not redefine a successful hit: the executed
full plant is still judged against the physical tolerance and cone. The current
3 mm position and 2 degree direction margins are small development buffers, not
experimentally calibrated uncertainty bounds. Before flight they must be
replaced by margins derived from held-out one-attachment model error and online
state uncertainty rather than tuned after observing task success.

The velocity is the tip velocity relative to the target: it is therefore the
world tip velocity for the present stationary target, not tip velocity relative
to the drone. For a moving target, the corresponding target velocity must be
subtracted instead.

The 20 m/s² development acceleration limit is intentionally aggressive while
the spatial excursion remains 0.15 m. It is an exposed vehicle constraint, not
a claimed flight capability; it must be replaced by the measured thrust and
attitude envelope of the selected quadrotor.

Drone safety is imposed over the complete prefix from the current state to the
candidate event:

```math
\max_{j\leq k}\lVert p_{drone,j}-p_{start}\rVert\leq R_{max},
\qquad
\min_{j\leq k}\lVert p_{drone,j}-p^*\rVert\geq r_{keepout},
\qquad
\max_{j\leq k}\lVert v_{drone,j}\rVert\leq V_{max}.
```

Acceleration is hard-bounded when controls are sampled. Candidate events and
candidate control sequences are ranked lexicographically by:

1. path-safety violation;
2. hit-constraint violation;
3. negative directed free-tip kinetic energy;
4. event time; and
5. normalized acceleration effort and acceleration-change regularization.

For terminal vertex mass `m_tip`, the energy proxy is

```math
E_d(k)=\tfrac12 m_{tip}[\max(0,d^{*T}v_{tip,k})]^2.
```

This is a directional terminal-energy proxy, not a contact-force or effective
impact-mass estimate. Contact mechanics are deliberately outside this phase.

Consequently an earlier miss cannot beat a later valid hit, and control
regularization cannot trade away a physical task or safety requirement. A
20-step horizon will often not yet contain the final strike, so the safest
best-progress plan is allowed to start and continue the maneuver. Unsafe or
late replans are rejected. Actual success is evaluated from the executed state
at every physics frame, not assumed from a predicted impact time. The scalar
cost is retained only as a readable diagnostic; it is not the decision rule.

In plain language, the controller must:

- achieve the requested metric hit, direction, and minimum directed tip speed;
- use the forward-injection/recoil action structure;
- remain inside the drone excursion, target keepout, speed, and acceleration
  limits throughout the maneuver;
- maximize directed free-tip energy among events satisfying those requirements;
- choose the earlier event only when the energy is tied; and
- use less aggressive and smoother acceleration only as a final tie-breaker.

The motion budget remains centered at the mission-start position, so a brief
large translation cannot be hidden by replanning. Every update solves a new
fixed-`N` horizon from the predicted activation state and executes only its
first `M` controls. After a hit or mission timeout, a bounded damped controller
returns the simulated drone toward its start; that return is not part of the
hit objective.

The UI starts with a longer IPOPT solve, then performs a warm-started bounded
IPOPT update per replan. It exposes `N`, `M`, iteration limits, convergence and
acceptable tolerances, solve-time limits, derivative step, rates,
regularization, and vehicle limits. It displays planned feasibility and, after execution,
measured hit feasibility together with position error, direction error,
directed world tip speed, excursion, clearance, and drone speed. The `.npz`
archive contains the complete state and command trajectory, fixed task
definition, all physical constraints and controller settings, plan diagnostics,
the six casting-action parameters, timing, and matched-model
identity. A target change starts
a new maneuver; it is not applied inside an active event horizon.

## Offline verified whip trajectory

Run:

```powershell
.\.venv\Scripts\python.exe -m research_tools.whip_optimizer
```

This offline solver removes the restrictive six-parameter casting primitive
and searches a smooth multi-frequency acceleration basis over a four-second
horizon. The basis starts from zero velocity, returns to the mission center,
and can express several forward/recoil pumping cycles without the random-walk
drift of unrelated acceleration samples. For the current isotropic cable and
stationary target, the teacher search is restricted to the vertical plane
through the start and target; the simulated cable itself remains fully 3-D.
A mass-conserving 7-node DER screens many trajectories efficiently. Its winner
only initializes a local search on the immutable 21-node fitted DER. Hit
feasibility and every reported metric come exclusively from a final 21-node
rollout using the same path, forward/recoil, hit, direction, and energy
definitions as the controller. The archive stores both search histories, the
full trajectory, controls, event metrics, both model identities, settings, and
random seed.

The default task is target `(0.48, 0, 0.90)` m from drone start
`(0, 0, 1.50)` m, with a `0.15` m drone-excursion limit, `0.05` m hit radius,
`35` degree impact cone, and `1.0` m/s minimum directed tip speed. These are
development constraints, not claimed flight limits. This fixed-target oracle
established the trajectory and verification contract used by the multi-goal
policy stage below.

## Goal-conditioned SAC policy

The active baseline trains directly from online simulator experience:

```powershell
.\.venv\Scripts\python.exe run_sac_training.py
```

The worker generates all physical interactions with the current policy. It
stores real transitions plus future achieved-goal relabels in one uniform replay
buffer. Previously generated target-specific MPC trajectories remain preserved
as historical experiments but are not loaded by this baseline.

The nominal simulation policy maps the complete simulated DER state and the
requested target event to one normalized 3-D acceleration command every 20 ms.
For `N` controller nodes, its target-aligned observation has `6N+22` values:
all node positions and velocities relative to the attachment, drone
displacement and velocity, target displacement, desired impact direction,
requested speed, active hit radius and direction cone, time remaining,
and the previous 3-D action. Full-state feedback is intentional for this
nominal-policy phase; a later flight estimator must supply a belief state from
sparse measurements.

The actor is a radial-tanh squashed Gaussian over the unit acceleration ball.
Two Q networks, target networks, entropy temperature, and a CUDA replay buffer
implement off-policy SAC. The simulated plant enforces the exact DER link
constraints. The environment checks the metric hit sphere, direction cone,
directed world-frame tip speed, target keepout, and drone speed at every physics
frame. There is no start-centered excursion limit. Acceleration and speed
remain bounded by the vehicle model.

The hit event is the first swept intersection of the cable tip and a metric
target sphere. That contact must satisfy the requested directed speed and
velocity cone. Cable extension is not an additional success condition; the
task is defined by target contact and impact velocity.

The UI first runs a fixed-goal control experiment at 0.80 m horizontal distance,
0.10 m below the initial drone, with a radial horizontal impact vector. This
deliberately isolates learning stability before a multi-goal claim is made.
Target and impact-vector ranges remain exposed by
`--target-distance-min/max`, `--target-height-min/max`, and
`--target-azimuth-min/max` plus the corresponding impact-vector arguments. The
required directed impact speed is 1.5 m/s by default.

At reset, the environment constructs an episode-fixed frame whose forward axis
points from the initial drone toward the target in the horizontal plane. Cable
state, drone state, target displacement, and actions are represented in that
forward/lateral/up frame. Thus yaw-rotated goals have the same policy
coordinates for the current isotropic cable and yaw-symmetric point-mass drone,
while gravity, vertical motion, lateral motion, and the DER itself remain fully
three-dimensional.

The 0.961 m cable means part of the sampled distribution lies beyond the sphere
that a straight cable can reach from its initial attachment. Evaluation labels
episodes by this exact initial-attachment distance and reports within-reach and
beyond-reach success separately. This label must not be interpreted as proof
that the policy whipped rather than translated the vehicle. SAC has a 0.25 m
target keepout and a squared drone-displacement penalty applied at a successful
impact, but it has no hard start-centered excursion bound and does not penalize
the maximum path excursion. A defensible beyond-drone-workspace claim requires
an additional evaluation with a hard excursion path constraint.

The SAC plant uses 15 DER nodes. A matched reachability audit gave impact errors
of 49.9, 44.0, 23.9, and 21.9 mm for 7, 11, 15, and 21 nodes respectively.
Fifteen nodes is the smallest tested grid that preserves nearly the full-grid
impact accuracy with a useful margin; seven nodes was only 0.1 mm inside the
50 mm hit tolerance. This resolution choice is empirical and task-specific.

The four-second episode has no artificial phase boundary and no positive
injection-energy reward. The dominant return is the valid-hit bonus. A bounded
tip-to-target potential and small time, effort, and safety terms provide
secondary shaping. The policy can therefore learn when to drive and reverse
from the physics instead of being assigned a release time. On a valid
hit only, a terminal regularizer penalizes drone displacement from its start:

```math
J_{return}=\lambda_r\left(\frac{\lVert p_d(t_{hit})-p_d(0)\rVert}{L}\right)^2.
```

It is not accumulated over the maneuver and does not constrain maximum
excursion, so the drone remains free to execute an aggressive forward/recoil
stroke.

This is goal-conditioned off-policy SAC with future HER. After random warm-up,
each update samples one uniform batch from the combined real/HER replay. Every
completed episode contributes future achieved tip position-and-velocity goals;
each retained prefix is rescored with the same swept-contact, vector-impact,
and safety contract as real experience. The agent uses a squashed Gaussian
actor, two Q networks, Polyak target networks, and automatic entropy tuning.
Observations use running normalization and critic hidden layers use LayerNorm.
There is no demonstration replay, behavior cloning, prioritized replay, critic
ensemble, or curriculum.

The policy runs at 50 Hz and the DDER plant at 100 Hz. The default discount is 0.998,
which preserves approximately the same discount per physical second as 0.99 at
the earlier 10 Hz rate. Time-integrated reward terms are scaled by the control
interval. The UI starts with a 100,000-transition random warm-up and a
500,000-transition finite run; both remain explicit experiment settings rather
than claims that this horizon is sufficient for convergence.

Periodic deterministic validation uses fixed seed `seed + 1000` and selects
the deployable checkpoint first by balanced success: the arithmetic mean of
within- and beyond-initial-reach success. Aggregate success, lower position
error, higher directed speed, and lower impact drone displacement are ordered
tie-breakers. The requested output path always contains that best checkpoint;
the newest policy state is retained separately with `.latest.pt`. The latest
policy is a diagnostic snapshot, not a resumable trainer state: replay memory,
optimizer state, environment state, and random-number-generator state are not
stored. A normally completed finite run also retains the historical
`.final.pt` diagnostic artifact.
If a successful policy remains degraded for 20 evaluations, training stops and
keeps the best policy. Both checkpoints store the observation statistics,
networks, entropy temperature, model hashes, settings, seed, transition count,
and evaluation.
Earlier scalar-action, endpoint-only, or curriculum checkpoints have a
different contract and are not evidence for this policy.

The desktop training dashboard is the normal interactive entry point:

```powershell
.\.venv\Scripts\python.exe run_sac_training.py
```

It launches SAC in a separate CUDA process and shows the task-level signals
needed to judge learning. The primary curve is the recent training success rate
against completed episodes. Deterministic checkpoint diagnostics report mean
free-tip error, directed tip speed, unsafe-episode rate, drone displacement at
impact, transitions, throughput, and the retained best checkpoint.
The dashboard also replays the first fixed deterministic validation target at
every checkpoint using the current policy. This trajectory is taken from the
validation batch already being evaluated, so the viewer neither adds a rollout
nor changes replay, rewards, random-number streams, or policy updates. Command-
line training remains headless unless `--progress-json --validation-preview`
is requested explicitly.
`Train until I press Stop` removes the transition ceiling and disables the
finite-run degradation stop. `Stop safely` is cooperative: it finishes the
current small CUDA operation, atomically saves `.latest.pt` and the JSON
history, and leaves the best validated policy untouched. The partial JSON log
is rewritten after every checkpoint validation, so an indefinite run remains
inspectable. A manually stopped run does not repeatedly consume the held-out
test set; only its fixed checkpoint-validation panel is used while training.
The dashboard intentionally has no Pause or Resume control because the current
policy checkpoint is not a complete SAC training state.

After training, the retained checkpoint is evaluated with independent fixed
test seed `seed + 10000`. The JSON log records both seeds, aggregate
success/error/speed/energy/drone-at-hit metrics, and separate episode counts and
success rates for targets within and beyond initial straight-cable reach. The
earlier retained fixed-task checkpoint and rollout belong to the obsolete
single-goal contract; they are not evidence of performance on this continuous
goal distribution or of flight readiness.

For repeated training runs, explicitly pass the same held-out target seeds:

```powershell
.\.venv\Scripts\python.exe -m research_tools.sac_training_worker `
  --seed 43 `
  --checkpoint-validation-seed 1042 `
  --final-test-seed 10042
```

Changing only `--seed` changes network initialization, exploration, online
targets, and replay sampling; the two explicit evaluation seeds keep checkpoint
selection and final comparison targets fixed.

The following table is retained as a historical prior-replay experiment; it is
not the active training method. The controlled four-versus-twelve comparison changed only the
verified prior; model, task, SAC settings, validation/test seeds, and the
256-goal final test were identical. The four-demo baseline combined the three
artifacts in `goal_demos` with `fixed_task_oracle_15.npz`; the twelve-demo run
used the three directories listed above.

| Verified prior | Prior transitions | Selected transition | Overall success | Within reach | Beyond reach | Mean minimum error | Directed speed | Drone displacement at hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 demonstrations | 28 | 340,096 | 36.3% | 55.3% | 4.2% | 244.6 mm | 2.00 m/s | 0.448 m |
| 12 demonstrations | 79 | 180,096 | 59.8% | 55.9% | 66.3% | 154.4 mm | 3.80 m/s | 0.673 m |

The improvement is concentrated in beyond-initial-reach goals, while the
larger impact displacement reinforces that this is a nominal coverage result,
not evidence of a hard-workspace whip. This is a paired single-seed comparison
of the complete prior intervention. The four overlapping demonstrations were
also re-optimized, so the table does not isolate demonstration count or
far-target coverage as separate causal factors.

The twelve-demo method was also run from three cold starts and evaluated on the
same 256 targets:

| Seed | Overall | Within reach | Beyond reach | Selected transition |
|---:|---:|---:|---:|---:|
| 42 | 59.8% | 55.9% | 66.3% | 180,096 |
| 43 | 59.0% | 59.0% | 58.9% | 380,032 |
| 44 | 48.0% | 34.8% | 70.5% | 140,032 |
| Mean +/- sample SD | **55.6 +/- 6.6%** | **49.9 +/- 13.2%** | **65.3 +/- 5.9%** | - |

Run the internal `research_tools.sac_multiseed` utility on the three JSON logs
to reproduce the saved
aggregate. It refuses to combine runs unless the model, full task, prior replay,
settings other than training seed, validation/test seeds, and target-stratum
counts match. Far-goal success is comparatively repeatable; near-goal behavior
and the magnitude of misses are not yet equally stable.

## Full-plant controller benchmark

Run:

```powershell
.\.venv\Scripts\python.exe -m research_tools.controller_benchmark
```

The benchmark uses the full fitted DER as the plant, the reduced DER as the
controller, and deterministic oracle full-state projection at every logical
replan. By default it runs both a 7/11/15/21-node controller-resolution study
and matched plus `±30%` `EI`/`Cb` mismatch. Select one with `--study resolution`
or `--study mismatch`. The saved JSON contains only the primary control
metrics: hit feasibility, impact error, directed speed, direction error, drone
excursion, controller solve times, and missed online deadlines. Startup solve
time is reported separately because the maneuver begins only after that solve.

This remains controller-development simulation, not closed-loop flight. Oracle
state feedback, the provisional two-holder parameter transfer, and the absence
of aerodynamic cable drag/model residuals mean the result tests architecture
and local mismatch sensitivity, not real-world control accuracy. Held-out
one-attachment data and ultimately OptiTrack flight provide those tests.

## Deferred hidden-model OptiTrack testbed

This adaptation experiment is retained as research code but is not the current
public online workflow. `run_online.py` now launches the matched-model MPC
baseline above. A separate public adaptation launcher will be restored only
after that baseline and its real-time measurement interface are validated.

The controller panel loads the nominal cable artifact and the frozen SAC policy.
The truth panel independently loads a second cable artifact and can apply declared
`EI` and `Cb` scale changes. The SAC checkpoint is validated only against the
nominal model; the hidden-model hash is never used to satisfy that check or passed
to the controller. Phase A keeps cable length, gravity, numerical resolution, and
target distribution compatible so that the experiment isolates material-model
mismatch.

The simulated OptiTrack frontend returns one perfectly associated marker position
at each controller material site. For the current checkpoint this is a 15-point
grid. Marker positions correct the nominal cable coordinates at every 50 Hz action
boundary, while the nominal DDER prediction carries the unmeasured generalized
velocity. Causal BDF1/BDF2 marker velocities are still recorded as independent
diagnostics and future observer/adaptation inputs; feeding them directly to the
existing policy would change the velocity semantics it saw during training. The
nominal model predicts between observations, and the pre-correction marker
innovation is the primary mismatch signal.

This remains an online-architecture testbed in simulation. It does not open a
live OptiTrack connection, infer marker
identity, or adapt the cable parameters or policy during a trial.

The UI renders hidden truth only for the viewer and scorer, the controller-owned
nominal belief, measured markers, the target, marker innovation, and tip-target
distance. Saved NPZ trials keep measurement inputs, the previous applied command,
the newly selected command, truth and belief trajectories in separate namespaces,
plus both model hashes and the policy/task provenance.

This first plant retains the SAC training assumption: the action is a translational
acceleration tracked by a fast drone inner loop. Cable reaction is not fed back to
the point-mass drone. A genuinely force-coupled plant requires a dynamic attachment
vertex, drone mass and thrust, and retraining the controller; it is a separate
physics milestone. Phase A also assumes perfect marker association. Real unlabeled
OptiTrack data will need a causal ordering/data-association frontend before it can
replace the simulated marker source.

## Tip-only parameter-identification benchmark

Run the first sparse-feedback adaptation milestone with:

```powershell
.\.venv\Scripts\python.exe -m research_tools.tip_adaptation_benchmark
```

This is a deliberately narrow synthetic benchmark. Every record begins from
the known, settled vertical hanging state. The hidden plant supplies only exact
timestamps, measured attachment positions, free-tip positions, and recorded
attachment commands to the estimator; no interior cable nodes or velocities
cross the measurement interface. The nominal and hidden plants both use one
fixed 15-node, two-substep, four-constraint-iteration DDER solver in float64,
so a parameter change cannot silently change the numerical method.

The command evaluates matched dynamics, `EI +20%`, `Cb -30%`, coupled
`EI +20% / Cb -20%`, and a stationary no-excitation record. The last case is a
required negative control: the Fisher-information gate must freeze the update
when the tip trajectory cannot independently identify both parameters. For
informative cases, a bounded two-parameter output-error solve estimates only
log `EI` and log `Cb`. It reports parameter recovery, initial/final tip RMSE,
sensitivity singular values, information conditioning, posterior intervals,
and runtime. Each dynamic case is then replayed without refitting on a
phase-shifted excitation, and both nominal and adapted tip RMSE are reported;
this separates parameter recovery on the identification record from predictive
improvement on an unseen maneuver.

The reported tip metric is RMS Euclidean position error, not coordinate-wise
RMSE. The default run injects no noise and is therefore an inverse-problem
correctness check; use `--tip-noise-mm 2` for a separate declared 2 mm noise
experiment. The exact float64 path currently takes tens of seconds for each
two-second informative record and is not yet suitable for in-flight updates.

One JSON artifact is written to
`data/drone_mpc/tip_adaptation_benchmark.json`. It includes the source and
reduced-model hashes, complete solver and estimator settings, deterministic
excitation seed, a hash of each sparse measurement record, and the five case
results. It also stores hashes of the runner and adaptation implementation so
results cannot silently outlive code changes. The active source artifact is
currently a provisional transfer from
the older two-holder fit, so this benchmark verifies the identification
algorithm under controlled model mismatch only. It is not physical evidence,
real-flight adaptation, a full cable-state moving-horizon estimator, or SAC
policy adaptation. The next estimator milestone must introduce a
low-dimensional hidden cable-state belief while retaining the same tip-only
measurement boundary.

## Residual-learning gate

A learned residual is intentionally not trained on the current two-held-end
recordings. Their boundary condition differs from the one-attached/free-tip
whip, so such a network could improve an offline loss while corrupting the
deployment dynamics.

After the new dataset exists, first freeze the identified `EI` and `Cb` model.
Then train one small, shared local graph network to predict a correction to the
dynamic-node acceleration from relative edges, curvature, node velocity,
material mass/length, and attachment acceleration. The attachment correction
is identically zero. Apply the correction after the physical force step and
before exact inextensibility projection, following the physics-plus-residual
structure of DEFORM without copying its larger task-specific architecture.
Training uses recursive multi-step full-state rollouts, take-level train/test
splits, and both whole-cable and free-tip loss. The required ablation is the
same controller with physics only versus physics plus residual. Online learning
will initially adapt only a low-dimensional residual gain or final layer; the
full network is not updated during flight.
