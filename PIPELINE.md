# Cable Twin pipeline

## Public applications

The supported workflow has exactly three interactive entry points:

```powershell
.\.venv\Scripts\python.exe run_offline_fitting.py
.\.venv\Scripts\python.exe run_sac_training.py
.\.venv\Scripts\python.exe run_online.py
```

They separate OptiTrack identification, nominal SAC learning, and
nominal-versus-hidden-model evaluation. The third application is currently a
simulated OptiTrack testbed with perfectly associated ordered markers. It is
not live hardware ingestion, online material adaptation, or force-coupled
flight. MPC, oracle generation, benchmarks, and legacy perception paths remain
internal reproducibility utilities rather than additional public applications.

The project has three code layers:

- `optitrack_offline`: the canonical cable-identification experiment;
- `cable_twin/shared`: constrained rod dynamics and shared data structures; and
- `cable_twin/online`: the ZED/PIDNet particle-filter prototype.

## Canonical OptiTrack experiment

### Measurements and data split

Each Motive take contains one rigid body whose pivot is placed at the physical
cable attachment and ten ordered markers, `cable1:c1` through
`cable1:c10`. Marker `c1` is nearest the pivot and `c10` is on the
mechanically free distal end. The observation order is pivot, then
`c1...c10`: eleven material sites. CSVs retain the original 100 Hz timestamps,
metres, world coordinates, headers, and Motive quality fields.

Dynamic takes are assigned before fitting:

- **Training** supplies every complete, clean, non-overlapping one-second
  rollout window with equal weight; and
- **Validation** contains independent motions and never selects parameters,
  bounds, windows, or optimizer settings.

There is no endpoint-frame calibration because rigid-body orientation and
terminal twist are not model inputs. Two-holder takes with `c1...c9` describe
a different boundary-value problem and cannot be reused.

### Reduced cable model

The cable is a naturally straight, homogeneous, bending-isotropic discrete
elastic rod. Its state contains ordered 3D vertex positions and velocities.
Segment lengths are hard constraints:

```math
\|x_{i+1}-x_i\|=\ell_i.
```

Only the measured attachment position `x0(t)` is prescribed. The attachment
tangent and roll are free, representing a hinged/swivel connection rather than
a clamp. The distal node is dynamic and has no prescribed force or moment; its
measured trajectory is used only in the objective and validation metrics.

The fitted parameters are:

- bending stiffness `EI` (`N m^2`); and
- objective Kelvin--Voigt bending damping `Cb` (`N m^2 s`).

For the stated circular, isotropic, zero-intrinsic-curvature model, bending is
independent of material-frame roll. Quasistatic minimization together with the
free-end zero-torque condition gives zero twist strain, so twist variables and
`GJ` are eliminated analytically. This is not a zero-`GJ` material model and
it does not restrict the centerline to a plane.

Gravity and measured cable/marker masses are fixed inputs. A mass-weighted
constraint solve enforces length and the single prescribed pivot. Ambient
drag, contact, attachment springs, intrinsic curvature, learned residuals, and
torsional damping are outside this baseline. Torsion must be restored for
anisotropic or intrinsically curved cables, imposed root roll, a distal rigid
body, or environmental moments such as frictional contact.

The eleven observations and simulation grid are separate. With the default 21
vertices, the pivot and ten markers map to nodes `0,2,...,20`; latent vertices
are never treated as measurements. A grid or cable change requires a refit.

### Identification

Each rollout is initialized from measured cable state and driven only by the
recorded pivot trajectory. Prediction error is evaluated at the flexible
markers, including the free tip. Preflight checks rigid-body quality,
discontinuous pivot/marker motion, measured chord length, and the correction
needed to initialize an inextensible state. Bad transitions split a take, and
takes without a complete usable window are excluded before optimization.
Residuals never select data.

Identification uses a deterministic two-stage bounded optimizer. A fixed-seed
scrambled Sobol design searches the logarithmic `EI`/`Cb` domain on a
take-balanced Training subset. Projected CUDA-batched Adam then differentiates
through complete one-second rollouts. Periodic full-Training evaluations alone
select the restored and saved parameters. The artifact records the data split,
source hashes, cable measurements, solver settings, rejections, candidate
design, optimizer history, full-set scores, per-take errors, and local
sensitivity. A cancelled run never replaces a completed artifact.

### Held-out validation

A Validation take is initialized once from causal measured history. Only the
attachment pivot is continuously prescribed:

1. **Attachment-only** supplies no later cable observations and is the primary
   open-loop test of free-tip prediction.
2. **Every 1 s**, **Every 5 s**, and **Every 10 s** apply complete cable-position
   observations at fixed intervals while prediction continues between them.

A periodic update conditions position, retains predicted velocity, and projects
that velocity onto the corrected inextensible state. It is not a restart. A
scheduled update is skipped if the complete measurement is unavailable. An
unreliable pivot may be held at its last trustworthy position and is reported;
the distal endpoint is never held.

Validation output records model/take hashes, correction and held-pivot frames,
overall and time-resolved marker error, and errors at fixed horizons after each
observation.

## Drone whip simulation and MPC

The internal `research_tools.mpc_gui` application loads the fitted-model artifact
and derives an explicitly recorded mass-conserving controller model (seven DER
nodes by default, unchanged `EI` and `Cb`). The initial cable hangs vertically
from a point below the drone. A world-frame acceleration-controlled double
integrator represents drone translation, while the one-attachment DER advances
the free cable. The current baseline assumes a faster low-level flight
controller tracks commanded acceleration and neglects cable reaction on the
drone; this approximation must be checked experimentally.

The software plant is the complete fitted 21-node DER and advances at a logical
50 Hz on a priority CUDA stream. MPC uses a separate mass-conserving seven-node
DER. At every replan, the current full cable state is interpolated in material
coordinates onto the controller grid, then projected onto its length and
velocity constraints. Controller `EI` and `Cb` can be scaled explicitly for a
controlled mismatch experiment; the plant parameters never change.

A separate worker performs CUDA-batched CEM updates every 0.40 s (2.5 Hz by
default); the previous plan remains active during every solve. Each replan
predicts the state forward to its activation time, then publishes a new plan
atomically. The horizon shortens against one absolute maximum event deadline,
preventing receding-target procrastination. The full fitted software plant may
run slower than wall time; the UI reports its real-time factor. In a physical
experiment, OptiTrack supplies the full plant state and only the reduced MPC
must meet the online deadline.

CEM searches a six-parameter target-aligned forward-recoil primitive rather
than an independent acceleration at every control knot. The parameters define
forward elevation, recoil-plane deflection, forward and recoil excursions,
reversal time, and motion duration. Two zero-terminal-velocity quintic sweeps
produce a deliberate forward/backward reversal; their 100 ms acceleration samples are
bounded before rollout. The waypoint construction is inspired by the compact
two-sweep action in Planar Robot Casting, while the forward-recoil action and impact constraints below
remain specific to the aerial-whip task. The fixed development excursion is
0.15 m, and the exact simulated path is independently checked against it.

The task is a constrained, maximum-directional-energy hit. At every physics frame
the controller tests the free tip against a metric target tolerance, a minimum
world-frame velocity component along the desired impact direction, and a
direction-cone half-angle. World tip velocity is the correct collision velocity
for the stationary target; it is not measured relative to the drone. Drone
keepout, maximum excursion from the fixed mission-start center, and maximum
speed are path constraints over the complete prefix to the candidate event;
acceleration is hard-bounded.

An impact is considered a whip only after the drone has moved at least 7.5 cm
toward the target and recoiled at least 7.5 cm from its forward peak. Thus the
controller must inject energy with an explicit forward/backward base motion;
monotonic translation of the cable tip is infeasible. The default target is
0.48 m horizontally from and 0.20 m below the initial drone, making the strike
predominantly horizontal. The present aggressive 20 m/s² acceleration limit
remains an explicit vehicle parameter and requires flight-envelope validation.

The MPC may tighten the position and direction-cone constraints by separately
recorded robust planning margins. The executed full plant is still judged
against the original physical limits. Present 3 mm and 2 degree development
margins are uncalibrated internal buffers; before flight they must be
derived from held-out one-attachment model residual and state-estimation
uncertainty. A replan must also activate at least one control interval before
impact, preventing a last-moment coarse-model update from replacing an already
feasible whip.

Candidate events and CEM candidates are ranked lexicographically by path-safety
violation, hit-constraint violation, negative directed free-tip kinetic energy,
event time, then normalized acceleration effort and control changes. Thus the
highest-energy valid strike is preferred, an early miss cannot beat a later
valid strike, and regularization cannot trade away task or safety requirements. When no
feasible event exists, the plan is explicitly reported as infeasible rather
than being called a successful whip. The bounded post-impact return controller
is separate from the hit objective.

The internal `research_tools.controller_benchmark` utility is the deterministic
control benchmark. It advances
the full plant in fixed logical time, projects the full state at every replan,
and compares 7/11/15/21-node controllers as well as `±30%` `EI` and `Cb`
mismatch with the same seed and target. It stores hit feasibility, impact
error/speed/angle, drone excursion, solve timing, and deadline misses in
`data/drone_mpc/controller_benchmark.json`. This isolates discretization and
parameter sensitivity, but its oracle full-state projection and provisional
two-holder model are not experimental validation. Held-out one-attachment
rollouts and OptiTrack flight experiments remain required.

The internal `research_tools.whip_optimizer` utility is the fixed-target offline
control milestone. It removes
the restrictive six-parameter motion primitive and searches a smooth
multi-frequency forward/recoil basis over a four-second horizon. A
mass-conserving 7-node rod is used only for broad screening; the selected
motion is refined and verified on the exact 21-node fitted rod. Task and safety
definitions are identical in both stages, but feasibility and reported metrics
come only from the exact model. That maneuver established the control contract
now reused for per-goal demonstrations and multi-goal policy learning.

The internal `research_tools.goal_demos` utility generates the immutable prior for the amortized
controller. Its twelve default anchors span near, nominal, and far horizontal
radii and four height offsets. Each target receives four deterministic optimizer
restarts. Only an exact active-controller rollout satisfying the metric hit,
directed-speed, direction-cone, keepout, acceleration, and speed contract may
be selected. Selection uses the oracle's canonical lexicographic rank among
feasible restarts; an all-infeasible target produces no artifact. Each saved
NPZ records all restart seeds, feasibility outcomes, scalar terms, and the
winner. The generator preflights every destination and never overwrites an
existing artifact.

The present twelve-demo prior is stored as six verified artifacts in
`goal_demos_expanded`, four in `goal_demos_escalated`, and two in
`goal_demos_replacements2`. The generation entry point is:

```powershell
.\.venv\Scripts\python.exe -m research_tools.goal_demos
```

A fresh default generation writes to `goal_demos_generated12`, not into the
completed three-directory experiment above, and refuses to overwrite files.

The internal `research_tools.sac_training_worker` module trains the amortized,
goal-conditioned controller. The
nominal target-frame actor observation contains the complete simulated DER
node state, drone displacement and velocity, target displacement, requested
impact direction and speed, active tolerance/cone, remaining time, and
previous action. The actor returns a point in the 3-D unit ball, scaled to the
physical acceleration limit. The vector environment applies one new policy
action every 20 ms, integrates the cable at 100 Hz, and checks swept tip
contact at every physics substep.

The ground-up UI baseline first uses one target 0.80 m horizontally from the
initial drone and 0.10 m below it. The worker can expand distance, height,
azimuth, and impact-vector ranges after this fixed-goal control experiment is
learned reliably. The episode-fixed target frame
rotates world positions, velocities, and actions into forward/lateral/up
coordinates. For the current isotropic rod and yaw-symmetric point-mass drone,
this gives the policy yaw equivariance without making the cable simulation
planar. Some sampled targets are farther from the initial attachment than the
0.961 m straight cable length. The environment records those episodes
separately from within-reach episodes.

There is no hard-coded injection/release clock. A valid impact can occur at any
physics frame. First swept contact with the physical target must meet the
requested impact speed and velocity cone. A bounded target-progress potential
and small time, effort, and safety terms shape exploration; there is no
positive injection-energy reward. This
lets the policy choose when to inject energy and reverse rather than forbidding
an early physical hit. There is no start-centered drone-excursion limit. At a
valid hit, a secondary terminal cost penalizes squared drone displacement from
its episode start, normalized by cable length. It is not accumulated over the
path and therefore does not prevent a strong forward/recoil stroke.

Consequently, the label "beyond initial cable reach" is only a geometric
partition of the sampled goals. It does not prove that success was produced by
a whip or that the target lies beyond a hard-bounded drone workspace: keepout
and the displacement penalty discourage direct vehicle motion, but neither is
a hard excursion constraint. That stronger claim requires a separate protocol
that enforces maximum drone excursion over the complete trajectory.

Policy training uses a 15-node DER plant. Under matched reachability searches,
7, 11, 15, and 21 nodes produced 49.9, 44.0, 23.9, and 21.9 mm impact errors.
The 15-node grid is therefore the smallest tested model that preserves nearly
the full-grid accuracy with a meaningful margin. Resolution was selected by
task dynamics rather than runtime alone.

Policy learning uses goal-conditioned SAC with future achieved-goal relabeling.
The worker stores real transitions and coherent hindsight episode prefixes in
one uniform replay buffer. A hindsight goal consists of a future achieved tip
position and velocity; reward, termination, and target-dependent safety are
recomputed under that goal. It uses a stochastic actor, twin critics and target
critics, and automatic entropy tuning. There is no demonstration buffer,
behavior cloning, prioritized replay, or curriculum.
The canonical command is:

```powershell
.\.venv\Scripts\python.exe run_sac_training.py
```

Periodic deterministic evaluation uses fixed seed `seed + 1000` and retains
the policy with the highest balanced success, defined as the arithmetic mean of
within- and beyond-initial-reach success. Aggregate success, lower position
error, higher directed speed, and lower impact drone displacement are ordered
tie-breakers. The deployment checkpoint stores that policy and its observation
statistics; a separate `.final.pt` artifact stores the last training state for
diagnosis, and `.latest.pt` stores the newest policy at every validation or
manual stop. These are inference/diagnostic policies, not resumable training
states: optimizer, replay, environment, and RNG state are not serialized.
Training stops after sustained post-success degradation rather than
overwriting a better validated policy. The retained deployment policy is then
evaluated with separate fixed test seed `seed + 10000`.
The optional `--checkpoint-validation-seed` and `--final-test-seed` arguments
decouple these target sets from the training seed for repeatability studies.

`run_sac_training.py` provides the focused desktop training workflow. It runs
the CUDA trainer in an isolated subprocess, plots fixed-seed checkpoint success
(overall/within/beyond reach) and mean tip error, and exposes cooperative Stop.
Finite training preserves the behavior above. Endless training runs until the
user stops it, disables degradation-based termination, saves best/latest
policies atomically, and incrementally persists the JSON history. A stopped
endless run does not run the held-out test automatically, preventing repeated
test-set use during interactive checkpoint selection.

A controlled nominal-simulation comparison held model, task, SAC settings,
validation/test seeds, and the 256-goal final test fixed. The four-demo prior
(28 verified transitions) selected transition 340,096 and achieved 36.3%
overall, 55.3% within-reach, and 4.2% beyond-reach success, with 244.6 mm mean
minimum error and 2.00 m/s mean directed speed. The twelve-demo prior (79 transitions)
selected transition 180,096 and achieved 59.8%, 55.9%, and 66.3% respectively,
with 154.4 mm mean minimum error and 3.80 m/s speed. Mean drone displacement at impact also
rose from 0.448 m to 0.673 m, so this result establishes improved nominal goal
coverage, not bounded-workspace whipping or flight readiness.
It is one paired training seed and changes the whole verified prior: the four
overlapping trajectories were re-optimized in addition to adding eight goals.
It therefore does not isolate demonstration count or far-goal coverage alone.
Scalar-action, endpoint-only, and curriculum checkpoints from earlier task
definitions are incompatible.

Three cold-start training seeds were then compared on identical target sets
(checkpoint seed 1042; final-test seed 10042). Overall success was 59.8%,
59.0%, and 48.0%; within-reach success was 55.9%, 59.0%, and 34.8%; and
beyond-reach success was 66.3%, 58.9%, and 70.5%. The corresponding mean +/-
sample standard deviations were 55.6 +/- 6.6%, 49.9 +/- 13.2%, and 65.3 +/-
5.9%. Thus far-target behavior is repeatable in this nominal simulator, while
near-target behavior and miss magnitude remain sensitive to SAC initialization.
The internal `research_tools.sac_multiseed` utility saves the strict
invariant-checked aggregate.

Earlier fixed-target results and checkpoints use a different task distribution
and do not establish performance for this multi-goal policy. Multi-target
simulation evaluation, a hard-workspace whip test, state-estimation error, and
real-flight robustness remain separate claims.

A learned dynamics residual is deliberately deferred until those new data
exist. The planned residual is a small shared local graph model that corrects
dynamic-node acceleration between the physical force step and exact length
projection. It will use recursive full-state training and remain only if a
held-out physics-only ablation improves both whole-cable and free-tip error.
The old two-held-end recordings are not compatible residual-training data.

This interface is also the boundary for later online material identification:
an updated `EI`/`Cb` artifact can be loaded between replans without changing the
state, control, or objective definitions.

### Ordered-marker hidden-model testbed

`run_online.py` is the Phase-A simulated deployment testbed. It loads a
nominal cable artifact and strictly compatible SAC checkpoint for the controller,
then loads an independent hidden cable artifact for simulation truth. The hidden
artifact can carry unseen `EI`/`Cb`; its identity and parameters never enter the
controller call. Both plants intentionally share length, gravity, controller grid,
and solver settings so parameter mismatch is not confounded with geometry or a
different numerical method.

The simulated OptiTrack boundary is a perfectly associated, ordered position at
every controller material site (15 for the current policy). Measurements correct
the nominal generalized coordinates at 50 Hz. The nominal DDER predictor supplies
generalized velocity and advances between corrections. Causal BDF marker velocities
are archived for diagnostics/adaptation but are not substituted into the existing
SAC, which was trained with DDER-state velocity semantics. The controller therefore
sees only measured marker positions, measured drone motion, its own nominal belief,
the goal, and previous action. Hidden nodes are available only after action selection
for rendering and scoring.

The first testbed remains an acceleration-tracked point-mass approximation and is
not force-coupled. It consumes no live OptiTrack stream. Real unlabeled-marker
deployment additionally requires causal marker ordering and track management.
Adaptation is deliberately absent in Phase A:
the paired fixed-nominal result establishes the baseline and records the innovations
that the later state/parameter adapter must explain.

### Known-start tip-only identification milestone

The internal `research_tools.tip_adaptation_benchmark` utility tests the first
sparse-feedback identification step
before any flight or policy update is attempted. It constructs independent
nominal and hidden one-attached/free-tip rods with an identical fixed numerical
configuration: 15 nodes, two substeps, four constraint iterations, and float64.
All cases begin from the known settled hanging configuration. The adapter is
given only timestamps, attachment positions, free-tip positions, and recorded
attachment commands; hidden interior nodes never enter its measurement API.

The deterministic suite contains matched dynamics, `EI +20%`, `Cb -30%`, a
coupled `EI +20% / Cb -20%` mismatch, and stationary motion. A bounded
log-parameter output-error solve estimates only `EI` and `Cb`. A local
sensitivity/Fisher test must reject an uninformative trajectory instead of
turning the prior into a fabricated update. The saved JSON records parameter
recovery, tip RMSE, uncertainty and conditioning, settings, random seed,
measurement hashes, runtimes, and source/reduced-model provenance. After
identification, a phase-shifted excitation is simulated without refitting and
the nominal and adapted RMS Euclidean tip error are compared as a held-out
prediction test. The default artifact is a noiseless same-simulator inverse-
problem check; `--tip-noise-mm` enables a declared noise experiment. Local
implementation-file hashes are stored with the artifact.

This milestone is synthetic known-start parameter identification, not a full
online moving-horizon cable-state estimator. It does not validate the current
provisional parameter artifact, demonstrate adaptation from physical tip data,
or modify the SAC policy. The present exact float64 solver takes tens of seconds
for a two-second record, so it is not yet a real-time estimator. Unknown-state
output feedback, accelerated inference, and then online policy adaptation are
later, separately evaluated stages.

## Compatibility

The legacy two-held-end dataset and twist-aware `EI`/`GJ`/`Cb` artifact cannot
identify or validate the new boundary-value problem. For controller development
only, the drone UI supports a narrow provisional transfer from the final v5
artifact: retain `EI`/`Cb`, eliminate `GJ` and terminal frames, and add the
measured marker mass at the free tip. The free-tip model must still be fitted
and validated from new one-pivot plus `c1...c10` captures before results are
reported.

## Image, online, and contact paths

The earlier PIDNet image-plane and ZED particle-filter code remains separate
from OptiTrack identification. After a compatible free-tip artifact is fitted,
the online path can reuse its `EI` and `Cb` with its own length, mass, grid,
and observation model.

Contact is a later extension. Cable capsules and rigid-object surface queries
must explain both cable deformation and object momentum. Surface friction or an
offset contact force can transmit torsional moment, so the torsion-free
reduction must be reassessed for post-contact prediction.
