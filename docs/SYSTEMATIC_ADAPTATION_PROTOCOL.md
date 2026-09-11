# Historical combined-fitting design, version 0.2

**Superseded as the active paper pipeline on 11 September 2026.** Read
[PAPER_PIPELINE_AUDIT.md](PAPER_PIPELINE_AUDIT.md) and
[PAPER_EXPERIMENT_PROTOCOL.md](PAPER_EXPERIMENT_PROTOCOL.md).
The selected clean-study baseline uses the implemented staged full-model fit;
the combined refinement below is an optional, unimplemented extension, not a
required stage or a description of the flown M1/M2 estimator. No implementation
or fitting authorization follows from this historical proposal. Exact prior
bytes are saved under `runs/audits/paper-pipeline-audit-20260911/before/docs/`.

---

# A systematic adaptation method for the clean aerial-whip study

**Current execution:** the user has authorized a refit of the existing staged
method, documented and frozen in [FROZEN_SYSTEM_IDENTIFICATION.md](FROZEN_SYSTEM_IDENTIFICATION.md).
Job `M2-frozen-refit-v1` uses that contract. The combined fitting stage described
below remains a proposed method extension and is not included in this refit.

Design version 0.2, 10 September 2026. **Implementation target, not a frozen paper
protocol or an implemented new fitter.** The user clarified that the current
preliminary/M0/M1/M2 runs are development work. After the method is refined, they
will collect a separate clean experiment for the paper. No fit, planner, promotion,
flight, deletion or data-role change follows from this document.

**User-defined paper focus:** the complete UAV system uses real-to-sim-to-real
model updates and MPPI to improve open-loop dynamic whipping maneuvers. The user
does not claim a new adaptation algorithm. Adaptation is a supporting system
identification step; ordinary refitting is acceptable. The priority is a reliable,
repeatable procedure and evidence that the complete system improves. The whip's
PVA sequence is open loop with respect to cable/task feedback; onboard drone
tracking control remains active.

**Historical v0.2 proposal:** regularized refitting initialized
from the previous model, with fixed training-data replay and a complete-trajectory
objective. Implement this as one reusable update procedure for every round.

| Step | Fixed rule |
|---|---|
| Prepare | Same command/coordinate/mask checks; causal cable history 1 s and drone history 0.4 s; whole planned whip interval |
| Reuse data | Eligible new and prior training takes only; newest/older/preliminary family masses 1/0.5/0.5, normalized over existing families; preserve all validation roles |
| Initialize | Parent nominal parameters and both residual networks; new optimizer state for a changed dataset; retain measured geometry/masses |
| Identify components | Drone nominal/NN/attitude from recorded commands and pose; cable coefficients/NN from measured attachment and cable observations |
| Refine composition | Same combined drone, measured-boundary cable and command-driven cable objective every round, with parent/residual regularization |
| Stop and evaluate | Best training checkpoint, practical plateau, full temporal numerical checks, then fixed separate validation and retention report |
| Plan next | Eligible updated model drives MPPI under the study's unchanged task/reward/search protocol; freeze commands and forecast before flight |

This is weighted batch refitting from an inherited model, not a newly invented
online adaptation rule. The added combined objective is justified by the current
composition error, not by a need for algorithm novelty. It still needs numerical
implementation and development verification. Current M0/M1/M2 artifacts remain
unchanged. Freeze final numerical settings after this verification and before
the clean paper collection; do not tune them separately at each clean-study round.

**The two phases.**

| Development, now | Clean paper experiment, later |
|---|---|
| Use existing recordings to inspect failure modes and compare objectives, residuals, optimizers and stopping rules | Freeze those choices before collecting the new experiment |
| Preserve current model versions and original evidence | Use a new experiment identity, fresh raw recordings and explicit model ancestry |
| Existing validation takes can inform method development; keep their recorded fitting roles intact | Assign training/operational-validation/final-test roles before seeing outcomes |
| No current result is required to enter the paper result tables | Report every attempt and every update under the frozen protocol |
| Learned current checkpoints are development artifacts | Fit a fresh M0 from new preliminary training recordings; reset learned residuals and optimizer state |

Architecture and hyperparameters selected in development may carry into the clean
study. Current learned weights, fitted response gains and residual checkpoints
must not silently become the supposedly fresh M0. Its structural prior should use
declared engineering values and newly verified geometry/mass. Any pilot-informed
prior must be explicitly named if deliberately retained. Fresh recordings on the
same hardware establish a fresh experiment, not generalization to unseen hardware.
No present data should be deleted as part of this separation.

**Position relative to published work.**

| Primary source | Relevant precedent | Consequence for our method |
|---|---|---|
| [SimOpt, Chebotar et al.](https://arxiv.org/pdf/1810.05687), §III | Iterates real rollout collection, simulator-distribution adaptation and policy training; uses a constrained distribution update | A sim–real–sim loop is established work. Ours is a regularized point-estimate model update with offline trajectory planning, not SimOpt's distribution learner |
| [Mamedov et al.](https://arxiv.org/html/2407.03476v1), §§3–4 and Appendix C | Learns DLO dynamics from boundary motion with a regularized rollout objective and explicit initialization; uses staged initialization before full optimization | Measured-boundary cable fitting is a sound component identification problem. We additionally need to predict that boundary from drone commands |
| [COMPASS, Huang et al.](https://proceedings.mlr.press/v229/huang23c/huang23c.pdf), §§3–4 | Uses factorized trajectory discrepancy and learned parameter-to-discrepancy relationships; evaluates alignment and task outcomes | Keep drone/cable diagnostics and physical task performance separate. Our fixed cascade and diagnostic swaps are not learned causal discovery |
| [BayesSim, Ramos et al.](https://www.roboticsproceedings.org/rss15/p29.pdf) | Infers distributions over simulator parameters | A fitted point estimate and a local gradient check do not establish unique physical parameters or a calibrated posterior |
| [IRP, Chi et al.](https://arxiv.org/abs/2203.00663), abstract | Learns effects of action changes relative to an observed trajectory and demonstrates rope target whipping | Improved successive strikes alone do not establish the advantage of an explicit reusable dynamics model |
| [Wiggle and Go!, Jakobsson et al.](https://arxiv.org/abs/2604.22102), abstract | Uses task-independent rope identification to inform several manipulation tasks | Reusing identified rope properties across tasks is also established; our evaluation needs to address the aerial actuation and cable combination |
| [DEFORM, Chen et al.](https://arxiv.org/abs/2406.05931), abstract | Combines differentiable rod modeling and learning, with prediction and planning applications | Differentiable rods plus learning are not themselves a new contribution |

These papers supply established building blocks and related system comparisons.
Our study applies system identification within the full UAV whipping pipeline;
it makes no new-adaptation-algorithm claim. The relevant system evidence is whether
measured model updates and MPPI improve complete prediction and subsequent physical
whips, with data and computation costs reported. Transfer tests support any added
claim about new target conditions. Do not create an adaptation-algorithm benchmark
requirement that is absent from the user's paper scope.

**One model family.** Let theta contain the nominal drone-response parameters and
cable coefficients; phi contains the two residual networks. The forward model is

`logged PVA → loaded-drone pose response → rotated attachment → cable motion`.

The attachment is `p_a = p_o + R r_oa`. The drone component predicts imperfect
execution of commands. The cable component predicts motion under that execution.
The drone nominal parameters, cable nominal parameters, drone residual and cable
residual all receive the same declared update treatment each round. A parameter
need not change just because it is eligible for fitting.

Mass, lengths, marker placement and attachment offset remain measured structural
quantities. The current aircraft response is an effective model of the loaded
system, with no explicit cable-reaction feedback. Do not describe it as a fully
bidirectional rigid-body/rod model or add a second cable-load term without revising
and validating that model class. Residuals represent effective discrepancies;
they are not unique physical force measurements. Their inputs exclude target,
success flag, take ID and arbitrary trajectory phase.

Keep the current model class during the first development comparison. If a
residual fails to provide repeatable predictive benefit, select a simpler fixed
architecture before the clean study. The paper does not need both networks to be
necessary; it needs evidence for whichever architecture is used.

**One adaptation objective.** For a take i, let u_i be its recorded commands,
h_i its causal observation history and y_i its measured pose/cable observations.
The model rollout starts once from `I(h_i; theta, phi)` and runs without measured
state corrections over the scored interval. The initializer uses the same method
and observation history for all models; parameter-dependent latent states need
not be numerically identical. No future measurement may improve a claimed
prospective initial state.

Define three normalized, robust trajectory errors:

- `L_d`: commanded rollout versus measured drone position and orientation.
- `L_c|meas`: cable rollout versus measured markers, driven by the measured
  rotated attachment. This constrains the meaning of the cable subsystem.
- `L_c|pred`: cable rollout versus measured markers, driven entirely by the
  predicted drone attachment. This is the complete prediction used for planning.

Use the fixed composite objective

\[
J_k(\theta,\phi)=\sum_{i\in\mathcal D_{\le k}^{\rm train}}w_i
\left[L_{d,i}+\alpha L_{c\mid meas,i}+\beta L_{c\mid pred,i}\right]
+\lambda_\theta R_\theta+\lambda_r R_r+\lambda_\Delta R_\Delta.
\]

`R_theta` penalizes scaled parameter change from the parent, using log coordinates
for positive coefficients and a separate scale for delay. `R_r` penalizes residual
magnitude; `R_delta` penalizes change from the parent's residual output at the
same states. All are normalized over takes and valid observations. Geometry and
units remain fixed. Residual acceleration bounds are model regularization, not
measured actuator limits.

The measured and predicted-boundary cable terms share observations. This is a
deliberate composite fitting objective, **not a sum of independent sensor
likelihoods**. No maximum-likelihood, Bayesian or statistical-independence claim
follows from it. The measured-boundary term discourages the cable from compensating
for an inaccurate drone; the complete term supplies the missing system objective.
Neither gives a theorem of physical identifiability or future improvement.

A concrete first development setting is alpha = beta = 1, retaining the current
2 cm position/cable and 0.05 rad orientation scales. Use equal position/orientation
weight within `L_d`, and the current half all-marker / half tip weighting in each
cable term. These are declared engineering scales and weights, not measured sensor
standard deviations or theoretically optimal choices. Use the current smooth
robust penalty `rho(s) = 2(sqrt(1+s)-1)` on squared normalized vector errors.
Normalize each term by its own valid sample weights. Invalid observations are
masked, not fabricated. Check this setting on development data before freezing it.
Do not choose weights separately for M1, M2, individual takes or late error peaks.

The first comparison should keep velocity/acceleration as diagnostics. Derivatives
of tracking are noisy; a new derivative loss would be another method change that
must be tested and frozen in development. Raw differentiated acceleration must not
be treated as exact ground truth.

**Fixed data memory.** Use all eligible training data from the same experiment
under a declared weighting rule. One rule consistent with the existing workflow
assigns unnormalized masses 1 to the newest whip batch, 0.5 to all older whip
training batches together and 0.5 to preliminary training. Normalize over the
families that exist. This gives 2/3 new and 1/3 preliminary at the first whip
update; later it gives 1/2 new, 1/4 old and 1/4 preliminary. Within old whip data,
give each generation equal weight; within a generation, each take equal weight;
within a take, share its weight across eligible windows. More frames or overlapping
windows do not create more independent trials. No validation recording enters
training memory, including through a parent checkpoint.

**Fixed preparation.** Verify command bytes, coordinate transforms, tracking body,
hardware/controller identity and missingness. A miss remains useful training data.
Exclude only intervals failing predetermined measurement/model validity rules,
with reasons retained. Use free-motion data before physical contact; a virtual
target-sphere crossing is not a physical collision. Preserve raw global coordinates
and apply the rotated attachment offset consistently. Estimate clock alignment
from measured streams independently of a predicted trajectory. Do not allow
per-take time warps to repair a model's error.

Keep causal cable history 1.0 s and drone history 0.4 s for the first development
comparison. The strike interval is the whole planned maneuver, declared from its
frozen command package, with any modeled-domain exclusion documented. A 1.5 s
planning horizon is not a 1.5 s optimizer deadline or a mandatory flight duration.
Choose and record preliminary window lengths before the clean study; use identical
windows and masks across competing methods. Report complete free rollouts even if
GPU training internally uses temporal checkpoint blocks. Such blocks must retain
full derivatives and must not reset the physical state.

**Fixed solver procedure.**

1. Load the parent and the reviewed training manifest. Evaluate its full objective
   and component errors as the baseline. Preserve its files and checkpoints.
2. Obtain a staged initialization: nominal drone with inherited residual fixed,
   drone residual, attitude refinement, measured-boundary cable coefficients,
   then cable residual. This preserves the current useful subsystem structure.
3. Perform a final refinement of the assembled model against the same `J_k`.
   Nominal and residual blocks may use different numerical optimizers, but every
   final-stage accepted update and checkpoint is judged against that objective.
   Any discrete-delay proposal in this stage must also be judged by `J_k`, rather
   than silently changing delay using a position-only score.
4. Keep the parent and staged initialization among the checkpoint candidates.
   Select the best eligible checkpoint on training data only. Stop by one fixed
   practical-plateau rule, preserving best/current weights and optimizer state.
   Report manual stops, numerical failures and evaluation guards distinctly.
5. Freeze the candidate and its complete provenance before operational validation.

This is one algorithm applied to every update, including repeated data batches.
The exact final-stage optimizer schedule, scales, regularization strengths and
plateau settings must be benchmarked in development and serialized before the
clean experiment. They are not yet frozen by this design note. Existing staged
CUDA code and a differentiable execution path provide building blocks, but do not
establish that the new combined gradient/optimizer is implemented or fast.
Verify a full command-to-tip numerical derivative and production-rollout parity
before training. Batch independent takes/candidates on GPU; benchmark actual
runtime and memory. Useful gradients and physical accuracy are separate questions.

Weak sensitivity does not justify inventing parameter precision. Use regularization
and fixed physical-parameter sensitivity/profile reports. Flag ambiguous gains,
delay, stiffness and damping; do not manually unlock a different parameter subset
because a validation take looks unfavorable. An ensemble or learned parameter
posterior is optional later work, not necessary to call this procedure systematic.

**A fixed decision after fitting.** Separate training checkpoint selection,
operational acceptance and final paper testing. Operational validation can decide
which model to use next, but it is then part of the adaptation algorithm, not an
untouched final test. Predeclare acceptance before the clean experiment.

A conservative development starting rule is: complete-tip equal-take RMS must
decrease, while complete all-marker/pose errors, conditional cable error and
declared earlier-data retention metrics must not increase beyond predeclared
tolerances. Include a maximum per-take regression criterion so an average cannot
hide an arbitrary failure. Set tolerances from development repeatability and
measurement uncertainty, not from the clean validation result. Exact tolerances
remain an explicit freeze item; no numerical threshold is fabricated here.

If a candidate fails, retain the parent as the model used for planning, and record
the rejected candidate and reason. Do not change the weights, success gate or
threshold until it passes. Keeping a parent is a legitimate outcome; the algorithm
does not promise improvement at every round. Operational acceptance is not proof
of statistically significant or prospective improvement. No automatic model
promotion is enabled by this document.

**The clean experiment.**

1. Freeze a versioned manifest containing model class, structural priors,
   initialization, all losses/weights, replay rule, optimizer/stopping, acceptance,
   target definition, planner settings, seeds, data allocation and evaluation.
   Record the software revision, environment and complete configuration hashes.
2. Create a new experiment directory and collect fresh preliminary recordings.
   Split whole takes in advance. Initialize a fresh M0 without inheriting today's
   learned checkpoints or optimizer state. Reset neural output to its declared
   initialization. If hardware differs, measure and record its geometry/mass.
3. Plan with M_k and freeze the executable CSV, exact model and forecast before
   each real flight. Record all attempts. Score each original forecast before
   using that round's training recordings for the next update.
4. Apply the same frozen adaptation procedure to the preassigned new training
   subset plus training memory. Keep operational-validation data out of gradients
   and checkpoint selection. Record candidate acceptance/rejection and compute cost.
5. Repeat for a predetermined number of rounds or a predetermined stopping rule.
   Do not stop collecting because the plotted curve looks good. Outcomes may
   improve, plateau or regress; all belong in the result.
6. After the adaptation sequence is frozen, evaluate fresh repetitions and targets
   withheld from adaptation and method tuning. Do not refit on them or adjust the
   method after viewing them. A changed method starts a new protocol version and
   requires an appropriately fresh test.

Current 001/002/004 versus 003/005 roles remain unchanged. For new data, preassign
roles with a recorded randomized/block schedule rather than deciding after seeing
motion. Preserve exact hardware/controller identity and record operating conditions
actually available in the logger; do not invent battery measurements. Spread
baseline/adapted flights across session or battery-order blocks where practical.
If a command fails to execute as planned, retain that fact rather than selecting
only successful reproductions.

The future paper must distinguish two tests:

| Question | Controlled comparison | Primary evidence |
|---|---|---|
| Did the model improve? | Run each frozen model on the same fresh recorded commands, same causal history and masks | Paired per-take complete-tip/marker and drone errors; conditional cable error as diagnosis |
| Did adaptation improve the real task? | Plan for the same declared target set with baseline/adapted models using the same planner protocol, then fly each frozen plan | Hit counts, target distance, valid observed impact-speed proxy, execution outcomes and real-trial cost |

Changed command sequences are expected in the second comparison. They do not
isolate model prediction on their own; the first comparison does. Fix the MPPI
reward, success definition, task geometry, search/stopping rules and seed protocol
across the clean adaptation study. The desired soft speed bonus can be part of
that frozen objective. Changing it halfway through would be a separate experiment.
Use unchanged feasibility checks and report their meaning; the protocol does not
introduce a new hard command-acceleration or 4 m/s requirement.

The primary hit definition remains observed tip entry into the declared target
sphere, with coverage and execution qualification. Wave shape and impact speed
are additional outcomes. A 4 m/s smooth reward scale is not a success threshold.
Missing observations give unknown evidence, not automatic misses or successes.
Impact speed is a kinematic proxy, not measured contact force or impulse.

Use independent takes as the unit for paired errors, and session blocks where
dependence requires them. Report all individual values and aggregate uncertainty
at that level. A thousand frames are not a thousand experiments. Choose the final
repetition budget from development variability and the desired precision; five
repeats per condition are a practical pilot, not automatically enough for a strong
reliability claim. Test at least a trained condition and new target/motion
conditions if claiming transfer. Another hardware configuration is required for
an unseen-system claim. Multiple learning sequences are required to measure
variability of adaptation curves, not merely repeated flights of one final model.

**Supporting development checks and system evaluation.**

The central paper comparison is the whole system with a frozen initial model
versus the same MPPI system using models updated by the fixed procedure, along
with original-forecast errors through successive rounds. The following component
comparisons are development tools or optional ablations tied to a specific claim;
they are not mandatory evidence for a new adaptation algorithm.

| Method | What it tests |
|---|---|
| Frozen initial model | Benefit over doing no adaptation |
| Nominal-parameter adaptation with inherited residual weights fixed | Benefit of updating residuals, without confusing an update with removing the parent's network |
| Existing full staged adaptation | Benefit of the added complete-prediction objective over the current method |
| Proposed staged initialization plus combined refinement | Candidate systematic method |

Use the same initial package, training data, masks, parameter availability and
declared compute protocol. Report training time and data cost; an extra refinement
stage is extra computation. An equal-compute comparison helps distinguish a better
objective from simply more optimization. If the paper claims each residual is
individually necessary, additionally compare freezing each one separately. An
inference-only network removal is a sensitivity test, not a substitute for a fair
trained ablation. These experiments are proposed, not launched by this document.

Action-correction or cold-start cumulative-refit baselines are optional when they
answer an explicit system claim about efficiency or an alternative control approach.
Do not require warm-start refitting to outperform a new-from-scratch fit merely to
use it as a standard identification component. Fresh-target tests are appropriate
if claiming transfer. PPO is a separate optional system comparison with its own
fair optimization and evaluation budget.

**Immediate development work.** Implement and numerically check the combined
objective inside a reusable adaptation driver, then compare it with the existing
staged method on today's recordings to verify this implementation target. Keep
both residual updates in the initial implementation; change the chosen architecture
only if development evidence warrants it. Finalize the remaining
manifest values and the flight budget only after that comparison. Then freeze
the method and begin the user's clean collection. A model-specific repair of the
current M2 is not the experimental definition.

Current evidence and implementation detail: [M2 diagnosis](M2_REGRESSION_ANALYSIS.md),
[completed staged M2](M1_TO_M2_ADAPTATION.md),
[model contract](ADAPTATION_MODEL_CONTRACT.md),
[evaluation infrastructure](SIM_REAL_EVALUATION.md). Those earlier documents retain
their historical experiments; this document defines the proposed study structure.
