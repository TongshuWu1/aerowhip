# Between-Trial Dynamics Adaptation for Open-Loop Aerial Whipping

Working research proposal, 5 September 2026. All algorithm choices and experiment sizes below are proposals, not established results. Publication suitability and novelty remain contingent on implementation, experiments and a final literature review.

Timing amendment: after this proposal was drafted, the user selected 20 Hz commands and 100 Hz physics with twelve internal DDER substeps. The one-second strike now has 20 XYZ actions (60 scalar commands). References to 30 Hz below describe the earlier design; they do not override the current saved configuration. The adaptation research direction is retained for later work.

## 1. Central question and scope

Can a small number of recorded aerial strikes improve a reusable drone–cable dynamics model sufficiently to enable fast, reliable correction of subsequent open-loop strikes, without repeatedly retraining a large reinforcement-learning policy?

The central object is the adapted dynamics model and the usefulness of its predictions for command correction. PPO/SAC supply an initial maneuver; comparing PPO against SAC is not the main scientific contribution.

Use both successful and failed trials. Failure is task information, not a physical-parameter measurement. Adapt between trials. During the strike, cable observations and hit information are logged but do not modify the force sequence or its cutoff. Onboard attitude stabilization remains active. Hover recovery follows the frozen cutoff. A safety intervention overrides execution but is recorded and counted as an intervention, never as successful autonomous completion.

Current configuration: initial drone and cable state; one-second maximum maneuver; 30 Hz world-force commands; 150 Hz physics; eight internal DDER substeps; 20 ms fixed nominal follow-through. These are operating choices, not proposed contributions. The real force-controller interface has not yet been verified.

## 2. Position relative to prior work

- [IRP](https://arxiv.org/html/2203.00663) already improves rope-whipping actions from earlier observed trajectories. Our method must show a benefit from retaining an explicit model, including transfer beyond the adapted strike.
- [DEFORM](https://arxiv.org/html/2406.05931v3) already combines differentiable DER, learned corrections and multi-step training. We must not claim that combination as new.
- [Learning on the Fly](https://arxiv.org/html/2508.21065v2) already couples residual dynamics learning and differentiable quadrotor-policy adaptation. Its feedback-control setting differs from frozen cable-strike execution, but rapid joint adaptation is established.
- [GenDOM](https://arxiv.org/abs/2309.09051) shows object-parameter-conditioned policies. This is an alternative to changing weights, and a useful optional baseline.
- [Wiggle and Go](https://arxiv.org/html/2604.22102v1) already reuses rope identification for multiple manipulation tasks. Our proposed addition is between-trial adaptation of a coupled aerial system and a validated, efficient correction mechanism.
- [AdaptSim](https://irom-lab.princeton.edu/AdaptSim/) shows why task-effective simulator parameters can differ from physically faithful parameters. We explicitly evaluate both prediction and downstream control, rather than assuming they coincide.
- [Gradient analysis](https://proceedings.mlr.press/v162/suh22b.html) motivates checking optimization gradients rather than treating differentiability as sufficient evidence of useful command updates.

Candidate contribution: an experimentally validated method that separates aircraft-response error from cable-model error, maintains a transferable hybrid model, and uses it to correct short open-loop aerial strikes with bounded computation and limited real trials. Each clause needs its own ablation or experiment. This sentence is a hypothesis about a defensible contribution, not a novelty claim.

## 3. Four hypotheses

H1: Separating measured-boundary cable identification from force-to-aircraft identification improves held-out coupled prediction relative to fitting both error sources indiscriminately.

H2: Updating identifiable physical parameters plus a restrained residual improves full-strike prediction beyond physical parameters alone, without degrading preliminary-data validation.

H3: Local force-sequence refinement using the updated model improves actual strike outcomes more quickly than fixed-model refinement and ordinary RL fine-tuning under matched data and compute budgets.

H4: A model adapted on one collection of strikes improves prediction and control for held-out targets/initial conditions without additional identification. A separate maneuver is necessary to support the stronger claim of cross-task reuse.

Reject or narrow the associated claim if its experiment fails. For example, if the residual does not improve validation, retain physics-only adaptation rather than forcing a neural-network contribution.

## 4. Data and measurement protocol

Each trial stores raw timestamped marker observations and validity flags, rigid-body position/orientation, estimator outputs and timestamps, the exact initial state used by planning, planned and sent force commands, controller telemetry when available, controller version, policy/model hashes, force convention and coordinate transforms, planned cutoff, contact evidence, and recovery/intervention outcomes.

Keep command, tracking and controller clocks aligned. Do not treat a sent force command as measured applied force. Model timestamps and input holds explicitly when reconciling 100 Hz OptiTrack, 30 Hz commands and 150 Hz simulation. Do not invent independent observations by upsampling.

Estimate initial position, velocity and cable shape from a causal pre-strike window. Offline smoothing may help identify dynamics, but a deployment estimator must not consume future strike observations. Learn/choose its window and filtering settings on development recordings.

Ten spatial markers constrain shape and motion but do not fully observe material twist. Do not identify an elaborate torsional model from these positions alone. Known masses, geometry and marker placement remain measured quantities unless an independent calibration justifies a change.

Separate datasets by complete trial/take, not adjacent overlapping windows:

- Preliminary fit/development recordings: initialize the model and retain broad motion coverage.
- Preliminary validation and protected test: retain their existing roles; do not access the protected recording for method development.
- Adaptation trials: sequential data available to each method under a fixed protocol.
- Development validation: choose hyperparameters and acceptance thresholds.
- Final flight/test trials: evaluate the frozen method after development. No subsequent tuning against those results.

Pre-contact motion is the default identification interval. Post-contact motion is excluded unless a contact model is explicitly included. Recovery footage is usable for compatible free-motion validation but must not leak across splits from its parent strike.

## 5. Model update

Use a state x containing aircraft translation, cable node positions/velocities and the actuator state needed to predict realized thrust. Let theta_c represent cable parameters, theta_a the actuator/controller model, and r_phi a small residual. Aircraft attitude must either be included in the model or represented by a validated effective force-response model; instantaneous arbitrary world-force realization cannot simply be assumed.

### 5.1 Separate the two identification problems

Cable identification: drive the cable boundary with the measured attachment trajectory, initialized from the measured full shape and velocity. Compare predicted versus measured cable markers. Include the body-frame attachment offset and attitude-dependent attachment motion. This reduces the opportunity for cable damping to absorb an aircraft tracking error.

Aircraft identification: characterize force-to-motion response with gentle tracking experiments, before adapting it from strikes. A minimal candidate model is a bounded gain, input delay and first-order force-response time constant, with mass/gravity fixed from measurement. Extend it only when held-out trajectories expose a repeatable missing effect. Cable reaction must be included for coupled strike replay.

Boundary-driven cable accuracy does not establish accurate cable reaction forces on the aircraft. Therefore the final test always replays the coupled system from initial state and recorded commands, with no measured-boundary forcing.

### 5.2 Fit identifiable quantities

Begin with a small set: effective bending stiffness, bending damping and, only if supported, a drag coefficient. The current fitted stiffness is very small; inspect profile losses and sensitivity before interpreting it as a precise material measurement. Optimize positive parameters in log coordinates and use documented physical bounds.

Inspect the sensitivity of predicted observations to each parameter, profile ambiguous parameters, and assess variability under resampling of whole takes. If damping and drag compensate for each other, fix one or report an uncertainty set rather than an unjustified point estimate. A finite Jacobian is not a proof of identifiability.

Per-trial initial state and timing offsets are nuisance quantities constrained by calibration/measurement uncertainty. Do not give each fit an unconstrained initial state or time warp that can conceal bad physics.

### 5.3 Objective and solver

Proposed objective:

L_id = sum over takes/windows/times/valid markers of normalized robust position error
       + lambda_prior * ||log(theta) - log(theta_previous)||^2
       + lambda_res * residual regularization.

Weight each take deliberately so a long recording does not dominate merely because it yields more overlapping windows. Normalize by valid observations and uncertainty. Position trajectories are the main supervision; derived velocity terms are optional and weighted by their uncertainty. Tip/end-region error can be reported separately without silently discarding the rest of the cable.

Start with short rollout windows and expand to the complete strike duration, always validating free rollouts. Use multiple shooting for optimization if necessary, with constrained continuity and initial-state variables; do not report frequently reset predictions as open-loop accuracy.

For low-dimensional physics-only fitting, use autodifferentiation with a bounded nonlinear least-squares/trust-region solver and several initializations. For the residual network, use a separate minibatch optimizer such as Adam. These are proposed choices for our implementation, not a reproduction claim about DEFORM. Benchmark against the existing fitting pipeline.

### 5.4 Neural residual

Fit physical parameters first. Train a small shared local cable network with relative geometry and velocity features, along with any physically justified boundary/material inputs. Do not feed the target position, success label, trial identifier or arbitrary trajectory phase into a reusable dynamics residual.

Predict a bounded correction to cable acceleration/force, or compare a constraint-aware integration correction as an ablation. Enforce or reproject the rod constraints consistently. Internal force corrections should respect equal-and-opposite reactions; aerodynamic corrections must be represented separately because they are external forces. A blanket zero-net-force rule would incorrectly prohibit unmodeled drag.

Mix preliminary and adaptation data. Penalize excessive residual magnitude and inspect stability, segment lengths and energy trends. Do not require mechanical energy conservation in an actuated, damped system. Decide whether to keep the NN from held-out long-rollout and transfer performance, not training loss.

Use an ensemble or fits from resampled takes to represent remaining model ambiguity. Treat this as an empirical uncertainty estimate until its predictive coverage is checked; do not call arbitrary parameter ranges a Bayesian posterior.

## 6. Fast adaptation of the next command sequence

For measured launch state x0, let U0 be the force sequence generated by the existing actor using the current nominal model. The simplest adaptation first replans with updated physics and the unchanged actor; this is an explicit baseline because changed simulated observations can already change its commands.

Then optimize U(z) = U0 + B z, where B interpolates a small number of correction knots into the 30 commanded force vectors. Start with four XYZ correction knots (12 variables) and compare a denser basis if needed. Every deployed command is held for one 30 Hz interval. The basis reduces optimization dimension; it does not reduce execution frequency.

Choose candidate contact/termination times on a predeclared bounded grid. For each candidate use a differentiable pre-contact loss on tip-target distance, desired directed speed, direction alignment, drone travel, and force/slew effort. Optimize smooth approximations, not the existing binary first-contact score. Use a trust region around U0 and evaluate the objective across initial-state/model samples. A sample-average loss plus a predeclared tail-risk term is one practical starting point.

Force magnitude, upward-force and rate limits must come from the verified controller/aircraft envelope. Parameterize or enforce them explicitly. Controller saturation must also be present in the rollout. A candidate is not accepted solely because the smooth loss improves.

Rescore every candidate using the unchanged strict first-contact, directed-speed, angle, non-tip-contact and recovery criteria. Check the complete trajectory for earlier invalid contact and repeated strikes. Use independent validation samples for the acceptance check; optimization samples are not sufficient evidence of robustness.

Freeze the accepted force sequence and cutoff before execution, with the configured follow-through and PID handoff. Reject nonfinite or infeasible candidates and retain the previous verified maneuver, or omit the attempt if none meets the launch criteria. Record the rejection rate because an approach that never executes cannot be reported as reliable striking.

### 6.1 Validate useful derivatives

The current inference planner is decorated with torch.no_grad and detaches commands. Implement a separate differentiable rollout for adaptation, then compare its forward prediction to the deployed inference path under identical inputs.

Check directional derivatives against central finite differences for physical parameters and correction coefficients across representative states and several perturbation sizes. Test away from hard contact first; assess sensitivity near switching events separately. Verify small descent steps improve the exact resimulated objective.

In a controlled physical study, compare predicted and measured trajectory changes for small feasible command perturbations, accounting for initial-state variation. This is stronger evidence for adaptation than trajectory fit alone. Include the extra perturbation flights in the total data budget.

## 7. What happens to the policy weights?

Do not make full policy retraining a prerequisite for every attempt. First demonstrate model update plus sequence correction. This adapts the commanded maneuver, not the actor weights; describe it accordingly in the paper.

If results justify an extension, generate corrected sequences for a diverse development set of initial states and train the actor to reproduce them, or fine-tune its weights through the differentiable model. Validate on unseen initial states. A direct sequence-output policy is a different architecture from the current nominal-state rollout actor; do not silently equate them.

Parameter conditioning, low-rank updates and full updates are alternative experiments, not three required components of the initial method. Avoid expanding the first paper until the core adaptation result is established.

## 8. Experiments and causal comparisons

### A. Identification and prediction

Compare initial model, updated physical parameters, physical parameters plus residual, and an ablation that fits cable/controller effects jointly without the measured-boundary separation. Report cable-only and full coupled predictions on held-out trials and prediction horizons. Report both tip and all-marker errors, time-alignment uncertainty, and parameter stability across data subsets.

### B. Between-attempt strike adaptation

Minimum comparisons:

1. Initial simulator and initial actor, no adaptation.
2. Updated simulator and unchanged actor, replanning only.
3. Fixed initial simulator plus the same sequence optimizer.
4. Updated physics-only simulator plus sequence optimizer.
5. Updated hybrid simulator plus sequence optimizer, if the NN passed validation.
6. A direct real-trajectory action-correction baseline, described as our local/IRP-inspired implementation unless the original IRP implementation is actually reproduced.
7. Ordinary PPO or SAC fine-tuning in the updated simulator under matched adaptation wall-time budgets; include it primarily in simulation if physical testing resources are limited.

Do not claim superiority to IRP from comparison with an arbitrary weaker substitute. Use common initial actors/priors, information access, real-data counts and target cases wherever possible. Count original CEM search and policy pretraining separately, including any extra data used to train an adaptation baseline.

### C. Reuse

Adapt on one target family, freeze the identified model, then optimize for held-out target directions/distances and different measured initial cable states without another physical fit. This establishes within-task transfer. If feasible, add a distinct task such as a controlled cable sweep through a waypoint with lower terminal speed; do not equate a new target with a new task.

### D. Stress cases

First distinguish simulation mismatches in cable stiffness/damping, actuator delay/gain and initial-state estimation. Then test modest, documented physical condition changes within a verified flight envelope. Avoid changing many hardware properties at once and interpreting improvement as successful identification of each one.

### E. Budget and statistics

Use an initial pilot to estimate variability and feasible trial counts. A provisional development design is five independent adaptation seeds in simulation, adaptation budgets of 0/1/3/5/10 physical trials, and repeated physical sessions for the selected comparison set. Final sample counts should follow feasibility and precision/power calculations; these values are not a guarantee of sufficient statistical power.

Pair simulation scenarios across methods. Randomize or counterbalance physical method order to reduce battery, wear and temperature effects. Reset methods to their designated initial model before independent adaptation sequences. Use a common logged-data benchmark to isolate model-update effects, plus on-policy adaptation runs to measure the complete interaction loop; these answer different questions.

Use trial-level confidence intervals for proportions and hierarchical/cluster resampling across sessions or adaptation runs when trials are dependent. Simulation seeds and individual frames are not independent real-flight replications. Publish all failures, interventions, optimizer rejections and right-censored attempts-to-success.

## 9. Measurement and paper figures

Primary outcome: valid first-tip strike meeting predeclared direction/speed conditions, followed by successful recovery. Also report hit alone. Report all attempts, including non-execution/rejection counts.

At 4 m/s, a marker travels about 4 cm between 100 Hz frames. Five-centimeter target geometry makes contact and pre-impact velocity measurement consequential. Use synchronized independent contact evidence or suitable high-speed imaging where feasible. Interpolation is an estimator, not ground-truth contact; quantify uncertainty and flag unresolvable events by a predeclared rule. Do not claim measured impact force or impulse from tip speed alone.

Suggested figures:

1. Full pipeline with a visible boundary between recorded data after each flight and the frozen force sequence executed during flight.
2. Held-out tip/all-marker error versus prediction horizon, with initial/physics-only/hybrid models.
3. Predicted versus measured trajectory changes under small command perturbations.
4. Success plus recovery versus real adaptation trials, showing uncertainty and all evaluation points.
5. Success versus total adaptation wall time, including processing, fitting, optimization and validation; separately report flight/reset time.
6. Impact-speed and maximum-drone-travel distributions, avoiding speed-only averages that hide low success rates.
7. Transfer matrix across adaptation and evaluation target/maneuver groups with the model frozen.

Use the same numerical timestep and event rules across the main algorithm comparison. Report physical and numerical timestep sensitivity separately. Save raw per-trial outcomes and vector PDF figures; avoid smoothing that hides collapses or selective best-seed plots.

## 10. Implementation order and decision gates

Gate 1 — Confirm Lee-controller force convention, gravity treatment, actual command bandwidth, attitude/position feedback and timing. Complete synchronized logging and measured force-response characterization before treating point-force simulation as a deployment model.

Gate 2 — Build immutable recorded-flight replay with measured-boundary and coupled modes. Demonstrate that timing/initialization errors cannot explain away the apparent dynamics mismatch.

Gate 3 — Perform synthetic mismatch-recovery and held-out physical-parameter adaptation tests. Proceed to NN residuals only if repeatable missing motion remains.

Gate 4 — Implement the differentiable correction path and pass forward/derivative/finite-difference checks. Measure runtime and improvement against fixed-model correction.

Gate 5 — Run a small development flight study; use it to freeze hyperparameters, event definitions, budgets and statistical protocol.

Gate 6 — Conduct final comparison and transfer experiments with frozen methods and held-out cases. Only then write claims about sample efficiency, real performance and reuse.

Current state: simulation training and open-loop execution exist; physical adaptation has not been demonstrated, the new differentiable sequence optimizer does not yet exist, and the real controller interface is pending. The project should be described as a promising experimental platform rather than an already validated adaptive system.
