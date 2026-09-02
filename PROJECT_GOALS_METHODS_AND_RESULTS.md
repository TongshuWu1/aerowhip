# Project Goals, Methods, Contributions, and Research History

Date: 2026-08-31

## Executive summary

This project studies dynamic manipulation of a deformable linear object (DLO)
by a quadrotor. The immediate task is a whip-like maneuver: use the UAV to
accelerate a suspended cable so that the free endpoint enters a target with
sufficient speed and the correct impact direction. The broader goal is an
experimentally grounded system that can observe the current UAV/cable state,
adapt a maneuver to that state and the requested target, and execute safely
without relying on an unrealistically simplified cable model.

The project now has two important simulation results:

1. A stabilized variable-duration CEM planner solves the production simulator
   very reliably. It achieved 251/256 first-seed successes and 252/256 with at
   most three seeds, with zero population-to-authoritative replay flips. Its
   median planning time is 34.61 s, which makes it an excellent offline
   optimizer and reference but too slow for state-fresh online deployment.
2. A sequential PPO policy trained completely from random initialization for
   1,001,472 episodes reached 469/512 = 91.60% deterministic success on a
   physically propagated nominal state-bank audit. It is the first strong
   from-scratch neural result in the project. A zero-training compiler audit
   showed that continuous 10 Hz feedback materially outperforms precompiled
   open loop under EI/Cb mismatch and post-start disturbances, so closed-loop
   PPO is now the selected learned architecture. It is still a canonical-target
   simulation result, not a general target policy or hardware-ready result.

Many unsuccessful branches were scientifically useful. They showed that the
successful open-loop action manifold is narrow, direct high-dimensional SAC
exploration does not find it reliably, coordinate regression can destroy
phase-sensitive solutions despite low action error, diffusion candidate
support is incomplete, and compact two-pulse primitives were too restrictive.
These negative results motivated the simpler sequential-control PPO task and a
reward that directly shapes the desired impact rather than imposing arbitrary
time or UAV-motion failure gates.

## 1. Project goal

The long-term intended pipeline is:

```text
measured UAV state + measured distributed cable state + target + physics theta
  -> fast state-conditioned maneuver policy/planner
  -> one physical attempt
  -> measured outcome
  -> update the physical model when needed
```

The system should ultimately answer three research questions:

1. Can a physically grounded UAV-cable model predict the short, aggressive
   transients that create a whip?
2. Can a planner or learned controller generate a task-successful maneuver
   from the current state quickly enough that the measured cable state is not
   stale by execution time?
3. Can real-to-simulation model updates improve future maneuvers without hiding
   model errors inside uncontrolled reward or policy changes?

The current work remains simulation-only. No claim of real-flight readiness is
made.

## 2. Scientific task

### Production planner success

The historical production planning contract requires all of:

- endpoint-target distance <= 0.050 m;
- directed endpoint speed >= 4.0 m/s;
- impact direction error <= 30 degrees;
- cable endpoint c10 enters first;
- maximum UAV displacement <= 0.50 m;
- maximum UAV speed <= 3.0 m/s;
- maximum command acceleration <= 20.0 m/s^2;
- finite complete rollout.

These gates are retained for CEM benchmarking and model/planner comparison.

### Current PPO task success

The latest PPO experiment deliberately separates task success from smooth
motion preference. The first cable marker entering the target consumes the
single attempt. It is successful when:

- the first marker is c10;
- entry distance is <= 0.050 m;
- entry directed speed is >= 4.0 m/s;
- entry direction error is <= 30 degrees;
- the rollout remains numerically finite.

There is no required strike timestep. The simulated episode lasts 10 s and the
policy may strike at any physics step. UAV displacement, UAV speed, command
effort, body-rate effort, and action changes are continuous reward costs and
reported diagnostics rather than binary PPO task-failure gates.

This distinction matters when interpreting results: the final PPO validation
passes the current task 10/10 but fails the older numerical-gate diagnostic
because mean maximum UAV displacement is 0.666 m, above the historical 0.50 m
planning gate.

## 3. Physical and numerical model

The selected scientific model is:

`MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`

The active simulation stack contains:

- attitude-coupled UAV dynamics;
- a causal learned UAV residual with fixed normalization and history;
- a 12-node discrete differential geometry cable model;
- remeasured cable geometry, masses, and rigid attachment;
- fitted bending stiffness `EI` and damping `Cb`;
- CUDA float32 execution;
- the PCG32 damping backend;
- three DDER substeps per simulator step;
- four cable position projections;
- deterministic production contracts for action decoding and batched rollout.

The remeasured-geometry refit selected:

- `EI = 9.22579757946e-05 N m^2`;
- `Cb = 0.00277045699567 N m^2 s`.

The PRE_MPPI model gate passed. The protected `fig8vertical_002` take was not
used for model selection or policy evaluation in the reported milestones.

## 4. Production open-loop command contract

Production CEM uses one normalized 49-D action:

```text
16 acceleration knots x 3 axes = 48
active maneuver duration       =  1
```

The duration lies in `[0.45, 1.80] s`. The active maneuver is followed by a
fixed 0.30-s analytic settle with continuous position, velocity, and
acceleration, then a stationary hold to the 2.40-s evaluation horizon. The
same normalized action, decoder, complete command, fixed 2048 numerical batch
contract, and authoritative evaluator are used for population evaluation and
final replay.

This contract fixed two important defects:

1. the old post-maneuver command abruptly changed nonzero terminal velocity and
   acceleration to zero;
2. population candidates could change through codec, duration, precision, or
   deployment replay paths.

After repair, a 256-action population-versus-batch-one diagnostic had zero
hard-gate classification mismatches.

## 5. Production CEM result

The final CEM configuration uses population 4096, 5% elites, full covariance,
4-20 iterations, and authoritative selection from the final top 32 actions.

Results:

- canonical seeds 42-47: 6/6 authoritative successes;
- 256-context benchmark: 251/256 first-seed successes (98.05%);
- with at most three deterministic seeds: 252/256 (98.44%);
- population PASS to authoritative FAIL: 0;
- median planning time: 34.61 s;
- p95 planning time: 105.46 s;
- median successful maneuver duration: 1.111 s;
- median hit time: 1.100 s.

Contribution: this establishes a reliable offline optimizer, a source of
verified successful actions, and a strong oracle for judging learned methods.
It also quantifies why amortization or a fast feedback policy is desirable:
the cable state may change substantially during a 35-s online optimization.

## 6. Learning and planning approaches tried

### 6.1 MPPI and early trajectory optimization

Early work implemented accelerated DDER/MPPI planning, numerical consistency
checks, model-mismatch studies, and observer/adaptation experiments. These
experiments established core simulator and batching infrastructure but were
superseded for the canonical whip by stabilized variable-duration CEM.

### 6.2 One-shot SAC

Several SAC variants attempted to map an 83-D context to a complete open-loop
maneuver:

- nominal one-shot SAC;
- revised terminal task reward;
- canonical-context progress curriculum;
- spectral exploration and additional entropy;
- feasible-support audit;
- local K=4 constrained SAC with separate safety critics.

The branch did not produce a scientific stochastic success. Broad sampled
open-loop actions frequently left the feasible tracking region. Once tracking
error became large, unsaturated feedback produced finite but meaningless
runaway trajectories. Local spectral exploration improved feasibility but was
too restricted to reach the complete task. The conclusion was not merely that
one hyperparameter was wrong: terminal sparse success and a narrow coordinated
48/49-D action manifold made random off-policy discovery inefficient.

### 6.3 Simple sequential SAC

A simpler 10-s sequential SAC controller was then implemented with 6-D actions
(local acceleration plus body rates), the full UAV/residual/DDER model, and a
transparent target reward. A detached three-million-episode run was monitored
and stopped after performance deteriorated. Plots and partial artifacts were
preserved. This reinforced that SAC was unstable for this task/reward pairing.

### 6.4 Figure-8 SAC

A separate Figure-8 SAC task reused the legacy online tracking reward. It was
stopped because it did not address the central short-horizon open-loop whip
question and continuous Figure-8 execution has a different task structure.
The code remains isolated for reproducibility; it is not the active path.

### 6.5 Deterministic CEM imitation

The first amortization pilot trained a deterministic network to regress CEM
actions. Normalized action error appeared small, but held-out physical success
was 0%. For a phase-sensitive cable maneuver, averaging between distinct timing
or action modes can produce an action that is close in MSE yet physically
invalid.

### 6.6 Conditional diffusion plus outcome scorer

A production 49-D conditional diffusion generator and physical-outcome scorer
were implemented with state-level train/validation/test ownership, fixed noise
banks, deterministic DDIM inference, candidate ranking, and resumable CEM
teacher infrastructure.

The initial 768-context teacher campaign was stopped after 42 contexts to
avoid spending many hours before proving architecture viability. Existing-data
studies found and fixed a concrete DDIM terminal-timetable defect: sampling
started too close to the terminal noise cliff, causing 99.35% of generated
coordinates to require clamping. The repaired timetable reduced clamp use to
approximately zero and passed tiny-data memorization.

Further studies showed:

- context and target conditioning were active;
- state conditioning was weaker and less accurate;
- a structured FiLM conditioner did not outperform the selected flat model and
  was not promoted;
- oracle success rose with candidate count, showing incomplete noise/candidate
  support rather than a simple decoder bug;
- even large candidate batches did not establish the desired robust online
  policy.

This branch provided useful infrastructure and a clear negative result, but no
diffusion policy was promoted.

### 6.7 Robustness-basin audit

All 252 exact CEM actions replayed successfully, but very small perturbations
often destroyed success. Cubic-spline approximations retained 0/252, while a
train-only MLP with very small coordinate error retained only 35.81% physical
success. This established that low action MSE is not an adequate objective or
metric for the maneuver.

Contribution: the problem is not only multimodality. Many nominal optimizer
solutions are knife-edge in action space, so future teacher generation should
prefer robust solution basins and hard-gate margins.

### 6.8 Compact flick primitives

To test whether a much simpler abrupt maneuver could replace the 49-D whip,
bounded CEM searched compact two-pulse families:

- shared-axis 5-D primitive: 0/6 seeds;
- independent-axis 7-D repair: 0/6 seeds and zero population successes across
  221,184 rollouts.

These results showed that the tested compact families lacked sufficient timing
and directional authority. Learning was not applied to a family for which the
production optimizer could not find a valid action.

### 6.9 Iterative residual policy

An IRP-style diagnostic learned local physical outcome changes and applied up
to five corrections around a complete maneuver. Using only existing data, it
improved state-disjoint development success from 16.22% to 43.24%; the best
practical point was three corrections. A targeted continuation collected
22,528 responses around failed initializer actions but reached only 37.84%
development success, so the earlier checkpoint remained selected.

This was evidence that iterative outcome-guided refinement can outperform
direct coordinate imitation. It was not a one-query controller and did not
solve sparse joint/edge state-target support.

### 6.10 Sequential PPO curriculum and final whip PPO

PPO was tried after SAC failed. An endpoint curriculum showed that PPO could
learn useful cable authority. The experiment was then redesigned carefully:

- full 10-s production UAV/residual/DDER rollout;
- sequential 6-D control every 0.1 s;
- position/velocity command re-anchoring every 0.01-s physics step;
- acceleration plus roll/pitch/yaw body-rate authority;
- single first-entry attempt;
- no arbitrary 30-70 step success window;
- no UAV displacement/speed/acceleration task-failure gates;
- random policy, value, and optimizer initialization;
- no CEM, imitation, demonstration, or previous PPO checkpoint.

The initial balanced reward over-penalized UAV displacement and used an overly
forgiving direction factor. At roughly 204,800 episodes, the deterministic
policy could enter within 23.9 mm with c10 first in 10/10 rollouts, but impact
direction was around 73 degrees. The old direction mapping still granted about
65% direction credit to this sideways entry.

The successful repair used:

- robust maximum-displacement weight 15;
- integrated displacement weight 1;
- a smooth sigmoid direction score centered on the 30-degree task threshold;
- unchanged progress, strike-quality, effort, single-attempt, and success
  semantics.

At 73 degrees, the new direction factor is approximately 0.03 rather than
0.65, while gradients remain smooth.

## 7. Final PPO result

Run:

`whip_ppo_directional_displacement15_v1/2026-08-31T104735.951178Z`

Training:

- requested episodes: 1,000,000;
- completed episodes: 1,001,472 (batch-aligned overshoot 1,472);
- total task successes: 100,051;
- cumulative success: 9.99%;
- final 10,240-episode rolling success: 74.14%;
- throughput: 155.15 episodes/s;
- elapsed training time: 6,454.7 s.

Latest saved deterministic validation at episode 972,800:

- current task success: 10/10;
- c10 first: 10/10;
- median minimum tip-target distance: 17.7 mm;
- median first-entry distance: 34.6 mm;
- median first-entry directed speed: 6.54 m/s;
- median first-entry direction error: 22.85 degrees;
- mean maximum UAV displacement: 0.666 m;
- mean maximum UAV speed: 2.37 m/s;
- mean maximum command acceleration: 10.0 m/s^2;
- numerical failures: 0.

For future runs, training success is plotted using a 10,240-episode rolling
window (five 2,048-episode collection batches), and deterministic validation is
scheduled after every collection batch. Checkpoints are saved every 10,240
episodes to avoid unnecessary disk writes.

## 8. What the PPO result proves—and what it does not

It proves that, in the frozen production simulator, a randomly initialized PPO
policy can learn a fast, correctly directed, tip-first whip for the canonical
target while responding to mild initial-state variation.

It does not yet prove:

- generalization to arbitrary targets;
- robustness across the full physically propagated state bank;
- robustness to physics-parameter variation;
- transfer to real sensing, state estimation, latency, or actuator limits;
- safety or feasibility under the older CEM numerical gates;
- real-hardware success.

The 10/10 validation set is small and intentionally mild. The result should be
treated as a strong method signal, not a final policy acceptance test.

## 9. Main project contributions

1. **A measured, frozen aerial DLO simulator.** The project integrates UAV
   dynamics, a causal residual, remeasured geometry, and a 12-node DDER model
   under explicit versioned model freezes.
2. **A consistent production action and replay contract.** Population planning,
   saved actions, decoding, terminal settling, numerical padding, and
   authoritative replay now agree.
3. **A reliable offline whip optimizer.** Production CEM establishes task
   feasibility across state/target contexts and provides a scientific oracle.
4. **Evidence about narrow maneuver support.** Controlled audits show why
   apparently low action regression error does not imply physical success.
5. **A documented negative-result chain.** SAC, deterministic imitation,
   diffusion, structured conditioning, compact primitives, and targeted
   residual refinement were tested and retired or bounded based on evidence.
6. **A successful from-scratch PPO formulation.** Removing the artificial time
   gate, converting UAV limits to smooth costs, allowing attitude-rate control,
   and sharpening impact-direction shaping produced strong canonical-task
   learning.
7. **Reproducible experiment infrastructure.** Runs save immutable configs,
   status, checkpoints, validation histories, source hashes, plots, and explicit
   protected-test/hardware exclusions.

## 10. Current active pipeline

```text
physical recordings
  -> deterministic processing
  -> decomposed model identification
  -> selected model freeze
  -> full production simulator

offline reference:
  current state + target
  -> variable-duration CEM
  -> normalized 49-D open-loop maneuver
  -> ACTIVE / smooth SETTLE / HOLD
  -> authoritative scientific evaluation

selected learned controller:
  measured current UAV/cable state + canonical target + remaining-step fraction
  -> sequential PPO query every 0.1 s
  -> acceleration xyz + body rates
  -> full production UAV/residual/DDER dynamics
  -> continuous state feedback throughout the single 10-s attempt

retained baselines:
  production CEM -> high-quality offline reference
  pure SAC -> stopped negative learning baseline
```

## 11. Recommended next scientific step

The most valuable next experiment is not another optimizer or network family.
It is controlled PPO generalization:

1. freeze the successful PPO architecture and reward;
2. expand training across physically propagated initial states;
3. introduce multiple targets while keeping state-level split ownership;
4. validate every collection batch on fixed state-disjoint contexts;
5. keep a sealed final test untouched until settings are frozen;
6. compare current-task success and continuous UAV-motion metrics separately;
7. only after robust simulation generalization, introduce theta variation and
   then real-to-simulation validation.

The current canonical checkpoint should be preserved as the baseline. Future
changes should be compared against it rather than silently replacing its task
or reward.

## 12. Repository organization

- `simulator/`: coupled production simulator and desktop UI.
- `fitting/`: model identification, fitting, and validation.
- `planning/`: production commands, rollout, metrics, CEM, replay, and model
  freeze contracts.
- `learning/`: active PPO/SAC diagnostics and stable policy interfaces.
- `experimental_data/`: data contracts and deterministic preprocessing code.
- `config/`: versioned model, task, planning, and learning configurations.
- `tests/`: numerical, physics, planning, learning, and GUI regression tests.
- `results/`: curated PPO, SAC, and CEM plots, logs, checkpoints, reports, and
  shared state/context inputs.

Raw/processed measurements and active model assets remain under `data/`.
Complete historical run trees, milestone reports, and retired source are kept
in the dated sibling archive
`particle_filter_cable_project_archive_2026-08-31`; they were moved rather than
destroyed. Git history also preserves tracked retired files.

## 13. Safety and data boundaries

- Production model modified during the final PPO experiment: no.
- New CEM solves during the final PPO experiment: no.
- Protected `fig8vertical_002`: not evaluated.
- Real hardware: not executed.
- Radio connection or motor arming: not performed.
- All reported policy results: simulation only.
