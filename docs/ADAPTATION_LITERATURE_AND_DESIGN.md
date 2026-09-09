# Adapting the drone–cable simulator from real flights

Implementation follow-up: the user subsequently authorized the optional residual extension and differentiable execution mode. See [implementation and verification](history/20260909-doc-cleanup/DIFFERENTIABLE_EXECUTION_AND_RESIDUAL.md), including a legacy damping-switch sensitivity discovered during full-whip testing and a separately versioned numerical regularization. The review below describes the original M0 and research rationale; no adp0 fit or deployment-model replacement has occurred.

## Recommendation

Keep the present architecture: an effective drone-and-controller response model, followed by attachment geometry, followed by the differentiable cable model, with a small residual in each dynamics block. Use the five current adp0 flights to update this model locally around the flown maneuver. Fit the drone against measured drone motion and fit the cable against measured attachment motion before evaluating their composition. Joint optimization is an optional later experiment, not the default first adaptation.

The literature supports learning physical parameters and model discrepancy from real trajectories, then improving commands in the updated simulator. It also offers a separate route: directly correcting the next action from the previous trial. These routes can complement each other, but improvement in hitting does not by itself demonstrate improvement in the simulator. Our first objective should be predicting the recorded misses accurately; the next objective is generating a better command.

This is a research and design proposal. No fitting, training, command regeneration, or changes to active physics were performed for this review. The selected PPO remains stopped. Evidence below combines primary papers with inspection of the current repository; recommendations and limitations specific to our experiment are identified as our analysis.

## 1. What other researchers actually adapt

The following papers are relevant for different reasons. A tabletop cast, a weighted rope manipulated by a rigid arm, and an aerial free-tip cable maneuver are not interchangeable experimental demonstrations. Numerical errors reported across these papers use different tasks, horizons, observations, and metrics; they should not become a leaderboard for our experiment.

| Primary source and version | What changes after observing reality | Relevance and experimental distinction |
| --- | --- | --- |
| [DEFORM: Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects](https://arxiv.org/html/2406.05931), Chen et al., 2024 preprint, revised March 2025 | Differentiable rod parameters and learned integration corrections, evaluated through multistep prediction. | Closest modeling precedent. Its setup holds both end edges with manipulators; our execution model has one positional pivot and a free tip. Its residual is broader than our damping-only residual. |
| [Iterative Residual Policy](https://irp.cs.columbia.edu/irp_2022.pdf), Chi et al., RSS 2022 | A learned predictor uses the previous outcome and a proposed action change to select the next action. | Direct rope-whipping precedent. This is action adaptation, not identification of residual forces. Its simulation training uses 54 million trajectories, so copying its learned machinery is not a small-data shortcut. |
| [Learning Dynamic Rope Manipulation Using Task-Level Iterative Learning Control](https://arxiv.org/html/2602.21302v2), Suresh and Atkeson, May 2026 preprint revision | A local model maps task-critical rope errors into spline-command corrections through a quadratic program. | Particularly relevant to our offline trajectories. It composes robot and rope models, but uses weighted rope ends and explicitly neglects rope influence on the geared robot arm. |
| [Wiggle and Go!](https://arxiv.org/html/2604.22102), Jakobsson et al., April 2026 preprint | A diagnostic motion feeds a simulation-trained parameter estimator; CMA-ES then optimizes spline trajectories. | Strong precedent for identification followed by offline optimization, including 3D striking. It uses a ball-joint rope model and added tip weights; its inferred descriptors are simulator parameters, not uniquely measured material properties. |
| [Real2Sim2Real for Planar Robot Casting](https://arxiv.org/html/2111.04814), Lim et al., ICRA 2022 | Real trajectories tune simulator parameters; simulated and real data support action prediction. | A concrete real → simulator update → new action loop. The cable moves on a table, so contact/friction identification differs from our free-flight problem. |
| [A Distributional Treatment of Real2Sim2Real](https://arxiv.org/html/2502.18615v4), Kamaras and Ramamoorthy, RA-L 2025, March 2026 revision | Bayesian parameter inference produces a distribution used for object-specific randomized policy training. | Supports retaining uncertainty rather than claiming one uniquely correct cable. It studies vision-driven DLO manipulation, not our aerial whip. |
| [Sim-to-Real of Soft Robots with Learned Residual Physics](https://arxiv.org/html/2402.01086), Gao et al., RA-L 2024 | Regularized trajectory-matching residual forces are estimated first, then a network learns to predict them. | Useful residual-learning alternative with sparse markers. Experiments concern soft beams/robots; inferred forces are latent corrections, not directly measured ground-truth forces. |
| [Learning on the Fly](https://arxiv.org/html/2508.21065), Pan et al., 2025 preprint, [RA-L 2026 author page](https://rpg.ifi.uzh.ch/lotf/) | Real quadrotor data update residual dynamics, followed by policy adaptation in a hybrid simulator. | Strong drone precedent, but actions are collective thrust and body rates rather than our FullState PVA references. It deliberately uses analytical-only surrogate gradients for policy updates. |
| [Closing the Sim-to-Real Loop / SimOpt](https://arxiv.org/html/1810.05687), Chebotar et al., 2019 | Trajectory discrepancy updates a distribution of simulation parameters between policy-learning rounds. | Supports iterative uncertainty-aware adaptation using a simulator without differentiating it. Its manipulation tasks are not pure-cable aerial whipping. |
| [Accurate Simulation and Parameter Identification of DLOs using DER in Generalized Coordinates](https://arxiv.org/html/2310.00911v3), Chen, Bretl and Pham, revised 2025 | Controlled shape and buckling experiments identify bending and torsional behavior separately. | A reminder that parameter identification needs appropriate excitation. Repeated positions during one whip cannot establish all material properties. |
| [DeformX](https://arxiv.org/html/2606.22116v1), Yang et al., June 2026 preprint | A rod solver and rigid-body simulator exchange motion and reaction wrenches through multirate co-simulation. | Relevant to genuine bidirectional coupling and a possible later Isaac integration. It is a different undertaking from hosting our existing model inside Isaac Lab. |

### Modeling and action correction answer different questions

DEFORM provides the clearest justification for combining rod mechanics, learned corrections, and multistep losses. Its study collects 350 seconds per DLO at 100 Hz, trains on one-second horizons, and evaluates five-second predictions without ground-truth resets. These are useful methodological comparisons, not a requirement that our local update must collect that much data. Our five repeated one-second whips offer much less diversity, which argues for conservative updates. [DEFORM](https://arxiv.org/html/2406.05931)

IRP and task-level ILC explain why repeated failures can improve a task even when the physical model remains approximate. IRP predicts the effect of action changes from previous outcomes. The task-level ILC paper retains an approximate rope model and corrects the trajectory near a critical task event. Neither result establishes that a better next action uniquely identifies the real physics. [IRP](https://irp.cs.columbia.edu/irp_2022.pdf), [task-level ILC](https://arxiv.org/html/2602.21302v2)

Wiggle and Go is especially relevant to the user's suggestion of replacing repeated PPO training with spline optimization. It separates identification from task execution and uses a low-dimensional spline search after observing a diagnostic motion. Its “zero-shot” task execution still depends on real probe observations and a previously trained identification network. For our deadline, the transferable idea is identification followed by trajectory optimization, not rebuilding that entire network. [Wiggle and Go](https://arxiv.org/html/2604.22102)

Distributional approaches address ambiguity: several simulator parameter settings may explain similar observations. Our practical approximation can be a small collection of plausible fitted models, checked for disagreement when evaluating a new command. Fits from five overlapping leave-one-flight-out datasets are not a calibrated Bayesian posterior, and their agreement cannot exclude a shared modeling error. [Distributional Real2Sim2Real](https://arxiv.org/html/2502.18615v4), [SimOpt](https://arxiv.org/html/1810.05687)

## 2. Precisely which system we are identifying

The current deployment route is:

**Virtual force sequence → virtual simulated trajectory → 30 Hz FullState reference → effective loaded-drone response → moving attachment → cable dynamics → predicted tip.**

The real flight receives the FullState reference, not the virtual force sequence. Therefore, the drone model should predict how the actual vehicle, onboard controller, estimator, and command delivery respond to those references. It is an effective closed-loop response model inside an offline planning system. Calling the planned maneuver open-loop does not mean the onboard position/attitude feedback is disabled.

Write u(t) for the commanded position, velocity, acceleration, and supported heading fields. Let x contain the tracked drone origin position, velocity, orientation, and angular response state. The fitted block is ẋ = f_D(x, u delayed; θ_D) + r_D, with the current residual applied only to translational acceleration. This is not a directly identified motor-thrust model. In particular, fitted feedforward gains should not be interpreted as physical thrust calibration constants or copied into firmware.

The cable boundary follows A(t) = O(t) + R_WT(t) r_T. Here O is the cf_7 tracking origin, R_WT rotates its tracking frame into world coordinates, and r_T is the saved tracking-frame attachment offset. The current offset is approximately [0.006655, −0.012874, −0.055] m. The separate flexible span from attachment to C1 is 0.063 m; it is not another rigid offset.

The cable model then predicts q and v from this boundary and the initialized cable state. In execution prediction, the root position is prescribed while the distal tip is free. Rotating the offset is essential even though this does not impose a rotational clamp on the first cable segment. See [geometry conventions](GEOMETRY_COORDINATE_CONVENTIONS.md), [drone response](../simulator/drone_pose_response.py), and [research pose prediction](../simulator/research_pose.py).

### The real physics is bidirectional

The drone moves the cable, and cable tension acts back on the drone. The current virtual force-planning model includes a shared dynamic root and cable reaction. However, the fitted FullState execution predictor first predicts drone motion and then drives the cable boundary; it does not feed a separately computed cable wrench back into the drone predictor. These are two different stages of the pipeline, despite sharing cable assets.

The latter cascade can work locally because the effective drone model was learned with the cable already attached. It absorbs the observed loaded response for its training motions. It is not guaranteed to remain accurate if a new maneuver produces substantially different cable tension at similar drone state and command. A cable-state-free drone residual cannot explicitly distinguish those cases.

For now, retain this effective model and test its local predictive validity. Adding simulated cable reaction to it without reformulating and refitting the drone block risks counting loading twice. A future mechanistic alternative would separate an unloaded controller/vehicle model from explicit cable force and torque, with a residual for what remains. Bidirectional co-simulation is supported by work such as DeformX, but migrating to it is not necessary for the first controlled adaptation. [DeformX](https://arxiv.org/html/2606.22116v1)

## 3. What our existing residuals can and cannot learn

The drone residual is a small two-hidden-layer network, width 16, producing three bounded acceleration corrections, currently ±0.5 m/s². Its 15 inputs describe position error, velocity error, commanded acceleration, predicted velocity, and frozen prehover compensation. It has no direct attitude correction and no cable-state input. An attitude prediction error must therefore be investigated in the nominal attitude response, frame mapping, initialization, and timing rather than assumed fixable by this NN. [Implementation](../simulator/drone_pose_residual.py)

The active cable network has two hidden layers of width 32. Although the implementation supports more than one mode, the retained asset selects the dissipative mode. On each free node and axis it produces a nonnegative damping rate γ bounded by 2 s⁻¹, with acceleration correction −γv in world coordinates. Its root correction is zero and the separate fixed external drag is zero. This retains the user's decision against an imposed 0.3 s⁻¹ drag term. [Active model](../config/research_30hz/model.json), [residual implementation](../simulator/cable/residual.py)

This is a useful constraint: the continuous correction term removes kinetic energy relative to still air. It is not a guarantee that the entire discrete constrained integrator is energy stable, nor a general model of wind. It cannot supply an arbitrary elastic or integration correction, and its correction vanishes at zero velocity. A network can be present and enabled while its output family is too restrictive for a particular discrepancy.

Consequently, our residual is not equivalent to DEFORM's broader integration corrections. Likewise, the residual-physics approach of Gao et al. permits learned forces beyond damping, although regularization is needed because sparse trajectories do not uniquely determine those forces. These are reasons to inspect the remaining error, not reasons to replace our residual immediately. [DEFORM](https://arxiv.org/html/2406.05931), [learned residual physics](https://arxiv.org/html/2402.01086)

My proposed order is: update identifiable nominal parameters, fine-tune the existing constrained networks, and inspect held-out rollout errors. If cable phase/shape errors persist under measured attachment motion, consider a small structured extension, such as local curvature-dependent elastic correction alongside damping. Evaluate that extension as a separate model variant. Preserve translation invariance, root constraints, and bounded behavior; do not add a time-indexed correction that merely memorizes this CSV.

## 4. Differentiability is useful, but the current fast path breaks the graph

The cable fitting implementation explicitly supports backpropagation through a multistep rollout. Its checkpointing saves activation memory while preserving temporal gradients. Measured attachment samples are its boundary inputs and interior cable observations are prediction targets; the cable is not reset to those observations after every step. [Differentiable fitting](../experimental_data/differentiable_fit.py)

The fast rehearsal path is different. ResearchPoseModel.predict is decorated with torch.no_grad(). ResearchPhysics initializes detached tensors, invokes the cable step with create_graph=False, and updates through no-grad calls. Therefore, we cannot currently claim that the complete deployed command → drone → cable prediction path supplies end-to-end gradients. This finding does not invalidate its forward predictions or the existing separate differentiable fitter. [Pose runtime](../simulator/research_pose.py), [cable runtime](../simulator/research_physics.py)

Before joint fitting or gradient-based spline optimization, build a dedicated differentiable composition using the same equations and command timing. Check forward agreement with the production predictor, finite-difference gradients for selected parameters and spline coefficients, and the treatment of command-event timing. Delays implemented with discrete packet selection require special attention; differentiability of the continuous equations alone does not make every scheduling operation smooth.

Frozen NN weights and blocked gradients through NN inputs are different choices. The former still permits sensitivity of predictions to the input trajectory. Learning on the Fly deliberately uses the hybrid model forward but analytical-only surrogate gradients for policy updates, illustrating that a gradient approximation can be intentional. If we adopt such a choice, label and validate it rather than presenting it as the exact derivative of our hybrid simulator. [Learning on the Fly](https://arxiv.org/html/2508.21065)

CEM requires forward evaluations and remains usable without an end-to-end gradient path. Differentiability is an opportunity for fitting and later optimization, not a prerequisite for improving the next offline command.

## 5. What the five current flights support

The policy-scoped adp0 contains five repetitions of the same supplied command, linked to retained rehearsal 20260908-203914-039721. The complete CSV lasts 11.2 seconds; its scored whip is 0–1 second, with the saved predicted strike at approximately 0.94 seconds. The remaining interval is recovery and hold. All five corrected OptiTrack takes cover the whip with all ten labeled cable markers finite; some recovery intervals contain gaps. [Intake evidence](../runs/audits/20260908-adp0-intake/README.md)

The user confirmed no contact or intervention in these five flights. Their failures are therefore valuable free-flight dynamics observations, not trials to discard for missing the virtual target. Preserve raw data and exclude samples only through recorded quality reasons such as missing or inconsistent tracking. The reported 1.82–1.91 cm alignment RMS is disagreement between measured position streams, not model prediction error or target miss.

There are nominally five seconds of whip observations across five trials, plus recovery observations. One hundred measurement samples per second improve temporal resolution; they do not create hundreds of independent experiments. Repeated flights expose variability and repeatable bias near this motion, but do not empirically establish how outcomes change under arbitrary command perturbations. The model supplies those sensitivities through its assumptions.

This supports a local M1 update, not a claim of globally identified drone and cable physics. Keep measured mass, lengths, marker layout, and tracking-to-attachment geometry fixed. The current 157 g drone and 18 g cable-assembly masses are measured totals, while their detailed cable/marker allocation remains a modeling approximation. With this dataset, large compensating changes in material parameters and NN corrections would be evidence of ambiguity, not discovery of the true cable.

Old recordings may remain embodied in the M0 initialization. If new training batches use only current adp0, say so explicitly while acknowledging that historical prior. Do not silently mix old maneuvers into the new split or destroy them; they preserve the baseline's provenance.

## 6. Proposed adaptation procedure

### A. Freeze data interpretation before fitting

Preserve the flown CSV, original saved prediction, raw logs, marker identities, timing alignment, masks, and model hashes. Separate CSV command time, logger receipt time, and measured-motion alignment. Do not shift time or world coordinates to make the tip agree with the target. A model delay and an unknown logging delay can compensate for each other; alignment uncertainty should be recorded separately and tested over its supported range.

Use the full command sequence as the external input, with the correct pre-command hold and command history. Do not reuse the legacy maneuver-only classifier or assume that the longer controller log and hand-trimmed OptiTrack log begin together. Extract observed initial pose and cable state from the available initial/pre-command observations; estimates used for a causal forecast must not incorporate later maneuver samples. Take002 has a short pre-command margin and needs a correspondingly explicit initialization uncertainty.

### B. Diagnose where the error enters

Run three comparisons with identical timing and units. First compare the original saved prediction with measurements: this is the actual deployment forecast. Second predict the drone with the exact command and measured initial drone state: this separates some launch-state mismatch from response error. Third drive the cable with the measured, rotated attachment trajectory and an observed initial cable state: this isolates cable prediction error conditional on the boundary.

The third comparison is not a complete drone–cable forecast because future measured drone motion is supplied. Report it as a cable-only diagnostic. Finally compose the drone prediction with the cable prediction, with no future measured boundary, to assess the complete model. Cable observations needed for offline identification do not imply that the flight planner must consume cable state online.

### C. Adapt the drone block against measured drone motion

Fit a bounded subset of effective position/velocity gains, feedforward gains, delay, and attitude-response parameters before enlarging a NN correction. Use position and rotation rollout losses as primary observations; velocity terms can help when their estimation uncertainty is controlled. Avoid treating twice-differentiated noisy OptiTrack position as a clean acceleration target. An SO(3) rotation error avoids artifacts from subtracting wrapped Euler angles.

Then fine-tune the small acceleration residual with shrinkage toward M0 and a penalty on correction magnitude. Keep inputs causal and do not add trial identifiers, absolute elapsed time, or the measured future cable trajectory. Within this stage, nominal parameters and residuals can be updated in a controlled alternating schedule, but compare against a nominal-only update to expose unnecessary flexibility. The output remains a loaded FullState response model.

### D. Adapt the cable block using measured attachment motion

First profile the sensitivity to EI and bending damping Cb with residual weights fixed, then evaluate a constrained residual update. If a wide range of EI values predicts almost identically, retain a prior value or report a range rather than selecting a dramatic change with false precision. Classical DER identification studies use different excitations for different properties; our position-only repeated whip is especially weak evidence for torsion. [DER parameter identification](https://arxiv.org/html/2310.00911v3)

Use a masked robust loss over all ten markers, with an explicit additional tip term. Normalize per flight and per phase so ten seconds of recovery/hold cannot overwhelm one second of whip. Fit observed cable motion, not desired target coordinates: the true tip miss is the measurement the adapted simulator should reproduce. Short training windows may aid optimization, but selection must include an uninterrupted complete whip rollout and a separate recovery assessment.

Check the nominal solver at a finer timestep on a few representative trajectories before attributing all discrepancies to missing physical forces. A residual can compensate for numerical error at one discretization and change its meaning when solver settings change. Keep the timestep, node layout, and constraint treatment attached to the fitted model version.

### E. Validate the composition before joint refinement

Hold out complete flights, not random frames or overlapping windows. A useful development analysis is five leave-one-flight-out fits, in each of which drone and cable updates, learned preprocessing, and model selection exclude that flight. Per-flight initial observations remain legitimate supplied conditions, provided no later held-out dynamics train the model. Predefine model variants or perform selection inside the training subset; using the outer results to choose a design makes those results development evidence.

Compare M0, updated drone only, updated cable only, and both updated, with the same inputs and initialization convention. A cable improvement with a measured boundary can disappear under a predicted boundary; the composed result must be checked directly. A component may remain unchanged if current data do not justify its update, while both residual architectures remain enabled.

Only consider constrained joint refinement if the separately fitted models leave a consistent composed discrepancy. Retain direct drone and cable observation losses, priors, and small parameter updates. A tip-only joint objective lets drone and cable errors compensate, obscuring which block is wrong. Even a jointly differentiable loss cannot manufacture identifiability that the observations do not contain.

### F. Improve the command after approving the model evidence

Freeze a candidate M1 fitted to all eligible current flights after development comparisons. Re-evaluate the exact old command to document improved prediction, then optimize a new spline or retrain PPO in M1 as a separate action-generation step. Given offline execution and the existing planner, a local CEM spline search is a reasonable first choice; it avoids requiring policy retraining for every model revision. Planar casting provides another precedent for simulator identification followed by action improvement. [Real2Sim2Real casting](https://arxiv.org/html/2111.04814)

Evaluate candidate commands through both model blocks and retain 30 Hz FullState semantics, continuous recovery, and the established flight envelope. Where plausible fits disagree, constrain changes around demonstrated commands or penalize that disagreement. Export the reference that was evaluated as the drone model's input; exporting its predicted response as a new reference would change the experiment and effectively apply the response model again.

The next adp1 flights should be prospective evidence. Their measurements can later become training data for M2, but cannot simultaneously be described as untouched tests of that same refit. An additional repetition of the old command helps assess M1's prediction; a newly optimized command tests whether model improvement transfers to action improvement. Exact flight scheduling remains a later experimental decision.

## 7. Evaluation that answers the research question

| Quantity | Definition and interpretation |
| --- | --- |
| Drone trajectory error | Root-mean-square Euclidean position error over the selected phase, plus rotation error reported separately. State whether initialization is the saved forecast or measured initial state. |
| Attachment error | Euclidean error after applying the rotated offset. This is the boundary error actually seen by the cable. |
| Cable shape prediction | Masked RMS over valid marker position errors; report both measured-boundary and predicted-boundary cases. Normalize so missing markers do not silently change a trial's weight. |
| Tip trajectory prediction | RMS and peak error between predicted and measured tip at matched timestamps through the whip. This measures simulator accuracy. |
| Scheduled-strike miss | Measured tip distance to the target at the original saved strike time, approximately 0.94 s for this command. This measures the flown task outcome at the intended time. |
| Closest approach | Minimum measured tip distance during the whip, together with its time. This cannot replace scheduled-strike accuracy. |
| Timing error | Difference between predicted and measured closest-approach times, with tracking/alignment uncertainty stated. |
| Strike quality | Tip speed and direction, and any saved first-contact/marker-order criteria needed for a virtual success classification. Being nearby alone need not satisfy the saved task. |
| Variability | Per-flight results and spread across the five flights, rather than treating correlated frames as independent trials. |

For RMS, use the square root of the mean squared Euclidean distance, in metres; it is not the ordinary average distance. Report whip and recovery separately rather than a single whole-CSV number. Because these flights intentionally avoided a physical target, describe geometric or virtual success, not validated physical impact.

## 8. What we can claim, and what comes next

The promising contribution is an experimentally supported adaptation loop for an aerial free-tip cable task with a realistic command interface: virtual-force planning, FullState execution, learned loaded-drone response, and cable prediction updated from real motion. Merely combining two residual networks or putting a differentiable rope below a robot model is not a novelty claim; the cited literature already covers related compositions and adaptation mechanisms. The paper should establish which errors each component removes and whether a frozen updated model improves a subsequent real execution.

The immediate implementation should therefore be a new adp0 fitting/evaluation protocol, reusing checked equations while replacing legacy data-selection assumptions. Its first deliverable should be the three diagnostic comparisons and flight-wise M0 errors, followed by bounded component fits and complete-model ablations. The historical bootstrap runner should not simply be pointed at the new folder: its legacy takes, phase assumptions, and selection windows do not define this experiment.

I recommend keeping both residuals, keeping geometry and measured masses fixed, fitting the two blocks separately first, and using their complete prediction to accept or reject the update. Preserve M0 and publish a distinct M1 with its evidence. The key test is whether M1 explains what happened and correctly predicts what happens next—not whether it can be tuned to redraw the desired hit.
