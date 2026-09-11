# Aerial Whipping for ICRA 2027: Related Work, Contributions, and Experimental Design

## 1. Recommended positioning

**Frame the paper as a physical robotics system for targeted aerial whipping through repeated model refinement and offline replanning.** The central research question is whether a reusable predictor of the loaded UAV and distributed cable, refined from a small number of flights, makes subsequent executable commands produce more accurate real tip motion. The contribution is the demonstrated capability and the evidence explaining when the complete loop works. A new adaptation algorithm, a new elastic-rod formulation, or a new MPPI derivation is unnecessary.

The literature makes the boundary of this claim important. Robot arms already perform free-tip targeting, iterative trial improvement, simulation calibration, and open-loop dynamic rope manipulation. Recent aerial work already models flexible cables, identifies their dynamics, and demonstrates cable-shape control. The useful distinction is therefore the combination of a **flying actuator with imperfect command tracking, a retained distributed cable, and prospective improvement of targeted whipping through an identified command-to-tip model**. This is a defensible research direction, not a verified priority claim.[^1][^2][^3][^4][^5]

The strongest paper would establish three linked results: the system produces real targeted whips; model updates improve prediction on common unseen commands; and replanning with the updated model improves later real target approach under an unchanged planning objective. Transfer to another target, without additional fitting, would provide especially useful evidence that the model is reusable rather than a correction for one recorded trajectory. These are recommended hypotheses, not results already established.

The development evidence is encouraging but narrower. On the same three new M2 recordings, postflight causal-history replay gives mean complete tip RMS of 14.86 cm with M0, 7.99 cm with M1 and 7.03 cm with M2. One take regresses from M1 to M2. These are matched diagnostics, not the original nominal-start deployment forecasts. None of the thirteen recorded development flights has an observed entry into the fixed 5 cm virtual target sphere. These results support further study of model refinement; they do not yet support a claim of reliable real-world hitting.[^30]

**Use MPPI as the main planner.** It keeps the scientific chain direct: refine model, optimize commands, execute, measure. PPO is a useful optional comparison, but it adds policy training and exploration as additional variables. It is not needed to establish the intended system contribution, and switching to PPO does not by itself solve uncertain launch conditions.

## 2. Scope and actual system

This assessment concerns the selected single-target whipping system and its implemented staged identification procedure. It does not revive swing-and-settle or multi-target experiments. Existing M0/M1/M2 recordings are development data; a clean prospective study remains separate. The selected design in the existing experiment protocol is a release candidate, with unresolved execution, initialization, and fresh-M0 consistency checks.[^29]

### 2.1 Physical and computational contract

| Element | Current development system | Scientific interpretation |
|---|---|---|
| Platform | 145 g UAV; 17 g cable/marker assembly; 2S battery | Report the loaded vehicle and measurement conventions |
| Cable | 0.9525 m; 10 moving measured sites; 12 simulated nodes | Distributed transient deformation, including marker mass |
| Planning start | Tracked origin `[0, 0, 1.255]` m; hanging stationary cable | A nominal preparation state, not each flight's measured state |
| Target | `[1.25, 0, 1.0]` m; virtual sphere radius 0.05 m | Define interception geometrically and distinguish physical contact |
| Command interface | Desired position, velocity, acceleration, yaw and yaw rate at 30 Hz | A command-to-executed-motion problem, not perfect attachment tracking |
| Predictor | Effective loaded-UAV response → rotated attachment → cable mechanics and residual | A cascade; no explicit cable reaction fed back into UAV dynamics |
| Numerical timing | 150 Hz outer simulation; eight cable substeps per outer step | Numerical resolution, not a measured physical bandwidth |
| Identification | Staged regularized simulation-error minimization | Standard nonlinear system identification |
| Search | 512 random samples/update, four proposal families, 10 control points, 1.5 s maneuver horizon | MPPI-inspired offline trajectory optimization |
| Selection | Feasible contacts ranked before misses; fixed score within each class | Success definition is distinct from shaping preferences |
| Execution | Frozen PVA whip plus generated recovery, return and hold | Open loop for the cable task; onboard UAV tracking feedback remains active |

The command frame is the tracked origin. The cable root is obtained using the orientation-dependent offset, `p_attachment = p_origin + R r_offset`, rather than treating the tracked point as the attachment or center of mass. The fitted aircraft response includes gains, an effective delay, hover bias, attitude response and a neural correction. Cable fitting covers effective stiffness, damping, external drag and a neural correction. The selected acceleration residuals are bounded at ±0.5 m/s² per component; this is regularization, not an identified motor limit.[^28][^29]

The cable load is about 11.7% of the aircraft mass. Its influence cannot be dismissed simply because the drone is heavier. Fitting the loaded aircraft can absorb some load-dependent behavior, but the current cascade does not establish transfer to arbitrary cable configurations or loads. A useful paper validates this approximation within a defined operating domain rather than describing it as a mechanically complete, bidirectionally coupled simulator.

### 2.2 Task definition and language

Define a successful virtual interception by the free-tip trajectory entering a fixed target region during a prespecified whip interval. Define physical contact separately if an actual object is struck. A high-speed pass close to the target is useful performance evidence, but it is not contact. The current recordings concern an observed virtual target, with no reported external cable contact.

“Whipping” can describe the transient preparation, reversal and distal motion visible in the experiment. To substantiate that interpretation, show a cable-shape sequence and marker velocities alongside the UAV reversal. Do not turn every visual characteristic into a mandatory success gate. Human-whip studies and robot demonstrations motivate these diagnostics, but they do not supply a universal bend, velocity or wave-propagation threshold.[^10][^11]

Avoid an undefined “beyond reach” claim. A free-flying UAV has no fixed manipulator workspace, and an inextensible cable cannot place its tip farther than its length from the instantaneous attachment. If reach extension is part of the paper, specify the UAV flight region or stand-off constraint and show the target relative to it. Being beyond the cable's reach from the initial hover pose is a narrower, measurable statement.

Use **tip speed at interception**, not “impact power.” Speed alone does not determine force, impulse or transferred energy; these depend on the contact and the effective participating mass. A soft preference for fast forward contact is sensible for the task, but it is not a measurement of mechanical power.

## 3. Closest-work comparison

The table identifies overlap, not a numerical leaderboard. Different robots, target geometries, sensors, training sets and evaluation metrics prevent direct ranking of published error or success numbers.

| Prior work | Task and embodiment | Model or improvement mechanism | Execution/evidence | Consequence for this paper |
|---|---|---|---|---|
| IRP, RSS 2022 / IJRR 2024[^1] | Arm-driven rope-tip targeting | Learned trajectory/action-change relationship across attempts | Real iterative manipulation | Trial improvement for whipping is established |
| Real2Sim2Real planar casting, ICRA 2022[^6] | Arm-driven planar cable casting | Simulator fitting from real trajectories, then simulated action learning | Real planar casting | The real–sim–real loop itself is established |
| Self-Supervised Free-End Cables, 2024 preprint[^7] | Planar free-end cable targeting | Learned dynamic manipulation model | Physical cable experiments | Free-tip targeting is an existing task family |
| Wiggle and Go!, accepted CoRL 2026[^2] | Arm-driven striking, lobbing and draping | Diagnostic excitation, rope descriptors, offline optimization | Open-loop real motions | Identification plus offline rope striking is very close prior art |
| Task-Level ILC, RSS 2026[^8] | Robot rope manipulation | Repeated task-error-based action improvement | Physical trials | A simpler command-correction alternative deserves discussion |
| DeformX, accepted IROS 2026[^9] | Arm-driven deformable manipulation and striking | Coupled simulation; learned actions | Open-loop hardware replay among demonstrations | PPO-to-open-loop striking is not a standalone novelty |
| Shen et al., T-RO 2025[^3] | Single UAV and free-ended flexible cable | Continuum/POD model, identification, offline reference planning | Real cable-shape-feedback NMPC | Aerial distributed-cable control is established |
| Gabellieri et al., IROS 2025[^4] | Single-/multi-UAV cable configurations | Simulation-error identification and model-based references | Real cable validation and feedback | Cable identification and new-motion validation are established |
| Rapuano et al., ICRA 2026[^12] | Aerial cable pick-and-place | Reduced continuum and hybrid dynamics | Numerical predictive-control evaluation | Compare against modern distributed models, not only pendulums |
| Learning to Throw, June 2026 preprint[^5] | Quadrotor with releasable tethered payload | Identified UAV dynamics, rope simulation, PPO | Real throwing experiments | Agile aerial sim-to-real manipulation is close prior art |
| Distributional Real2Sim2Real, RA-L 2025[^24] | Arm-driven whole-body DLO reaching | Parameter posterior inference and randomized PPO training | Real visuomotor deployment | An explicit real–sim–real DLO framework is already published |
| Proposed study | UAV-driven retained-cable free-tip whipping | Repeated refinement of executable-command-to-tip model, then offline search | Prospective physical evaluation still required | Establish the value of the complete aerial refinement/replanning loop |

### 3.1 Ground-robot whipping and iterative improvement

**IRP is the main earlier comparator for learning across attempts.** It uses the previous observed trajectory and a proposed action change to predict a trajectory change, supporting iterative real manipulation after simulation training. Its rope experiment is arm-driven and uses a low-dimensional action parameterization.[^1] Our intended distinction is a forward model that remains available for new command optimization. That distinction becomes convincing only if the refined model predicts or plans motions that were not used in its fit. It is incorrect to dismiss IRP as having no learned dynamics.

**Planar casting is a direct precedent for the loop.** Lim and colleagues explicitly connect real observations, simulator calibration and subsequent simulation-based action learning. Their surface-supported sliding/placement task differs from our airborne retained-tip strike, but the conceptual real-to-sim-to-real recipe is already explicit.[^6] The paper should cite this lineage early and focus its contribution on the aerial task and complete prediction chain.

**Free-end cable manipulation narrows the task gap further.** Wang and colleagues study surface-supported planar targeting with a weighted cable free end. They distinguish simulator fidelity from physical final-position performance and report that the best simulator match did not produce the best manipulation result.[^7] This makes a generic “using a flexible cable to reach a target outside the robot's immediate workspace” claim too broad. For our airborne task, better model prediction and better targeting must similarly be tested as separate propositions.

**Wiggle and Go! is particularly important recent work.** A diagnostic wiggle produces learned rope descriptors; an offline optimizer then generates open-loop motions for several dynamic tasks. Its evaluation includes unseen-motion model fidelity and an optimization-based identification comparison.[^2] This overlaps more directly than a generic deformable-object survey. Our repeated flight-data refinement and aerial actuator model must therefore earn their place through evidence, not through the name of the pipeline. Its reported frequency-correlation metric should not be described as pointwise trajectory agreement, and its task-specific outcomes should not be compared numerically to our 3D tip RMS.

**Task-level ILC represents an alternative scientific explanation.** Repeated task-error-based adjustment can improve a rope maneuver without establishing a reusable simulator.[^8] If our paper claims broader usefulness than one-trajectory correction, demonstrate reuse on a held-out command or target. A matched simple correction baseline is valuable if practical, but a full reproduction of every published learning system is not required for a focused systems paper.

**Fast striking and open-loop replay already occur in related work.** Zimmermann and colleagues optimize deformable motions using implicit integration, including tip targeting and speed objectives. DeformX includes simulation-trained motions replayed on a real arm without online cable-tip feedback, and a horizontal striking demonstration.[^9][^13] Neither an aggressive reward nor replaying PPO as an open-loop trajectory should be listed as an independent contribution here.

### 3.2 Aerial flexible-cable manipulation

**Shen and colleagues are the closest aerial free-cable predecessor.** They model vehicle–cable dynamics with cable reaction forces, reduce cable shape dynamics, and demonstrate real feedback control. Their dynamic window-crossing setup combines offline trajectory search with online tracking.[^3] This is not merely a suspended point-mass study. Our cascade is simpler in its mechanical coupling; its justification must be measured prediction performance and task usefulness.

**Gabellieri and colleagues already distinguish cable prediction from vehicle tracking.** Their real identification uses measured endpoint motion, followed by validation under robot-generated motions; cable-output feedback is also investigated.[^4] This supports our measured-attachment diagnostic, while clarifying its limitation: a cable can look accurate when driven by the true attachment even when the planner's command-to-tip forecast is poor. Complete prediction must remain the headline model metric.

**Continuum/hybrid aerial manipulation has progressed beyond swing suppression.** Rapuano and colleagues address attachment/detachment and pick-and-place with reduced continuum dynamics and hybrid predictive control; their reported experiments are numerical.[^12] It would be inaccurate to motivate our paper by saying aerial cable research only damps pendulum swing. The distinction is the physical retained-tip strike and its measured iterative refinement.

**Learning to Throw is a close agility and sim-to-real comparison.** It combines identified UAV dynamics with coupled rope/payload simulation and evaluates real PPO throwing. Its released payload follows a different task contract from our retained cable. Its main trajectory-optimization/MPC baseline is simulated, so the reported comparison must not be treated as a matched physical planner benchmark; its visual variant also uses projected keypoints in hardware-in-the-loop.[^5] Our physical baseline design should avoid the same comparability ambiguity.

### 3.3 Model refinement, hybrid dynamics and computation

SimOpt adapts simulator parameter distributions using real/simulated trajectory discrepancy, including robot and rope-related parameters. Our point-estimate, regularized staged fitting is not SimOpt, and updating two subsystems is not itself adaptation novelty.[^14] COMPASS and RAPiD further illustrate that “adaptation” can mean structured simulator adjustment or latent policy adaptation; our paper should state precisely which object changes between rounds.[^15][^16]

Kamaras and Ramamoorthy provide an explicitly named Real2Sim2Real DLO system: infer object-parameter distributions, train object-specific policies in randomized simulation, and deploy visuomotor behavior. Their target concerns the object's body rather than a tip-only strike.[^24] This is a close loop citation, even though the embodiment and inference method differ. Our point-estimate model does not represent a posterior distribution; its adequacy under launch and execution variation has to be established empirically.

Mamedov and colleagues' DLO identification work emphasizes structured dynamics and initialization; DEFORM combines differentiable rod mechanics with learning. Both are relevant foundations for hybrid prediction and its validation.[^17][^18] Our two bounded acceleration networks have their own architecture and regularization. They do not inherit another method's constraint handling, conservation properties or generalization guarantees merely because the model includes mechanics.

Discrete Elastic Rods is the appropriate mechanics foundation.[^25] The paper should specify which discretization, attachment, damping and constraint choices the actual simulator uses, rather than attributing all implementation details to that original formulation.

The same applies to the aircraft. Data-Driven MPC for Quadrotors learns residual aerodynamic effects for feedback control and explicitly tests unseen trajectories. NeuroBEM combines rotor/motor modeling with residual dynamics and evaluates both local predictions and integrated flight behavior.[^19][^20] Our model instead predicts the response of a loaded UAV under its fixed tracking controller. It should be described as an effective closed-loop response model, not an identified standalone aerodynamic vehicle model.

GPU batching makes the experiment practical, but parallel sampling is part of the established MPPI literature, and recent systems such as FLASH also emphasize fast GPU-based deformable learning.[^21][^22] Report measured runtime, hardware, numerical precision and workload. Do not infer a computational contribution or claim “real time” from GPU use alone.

## 4. Contributions and claim boundaries

### 4.1 Recommended contribution statements

The following are **proposed final-paper statements conditional on completing the corresponding evidence**:

1. **An experimentally demonstrated aerial whipping system** that generates executable UAV commands for targeted motion of a retained cable's free tip, using an identified loaded-vehicle response model and distributed cable dynamics.
2. **A repeatable model-refinement and replanning pipeline** that uses recorded flights to update the same model class and generates subsequent open-loop maneuvers under a fixed objective, with prospective evaluation of prediction and task performance.
3. **An experimental analysis of the complete prediction chain**, separating aircraft response, cable response under measured attachment, and full command-to-tip error, with a held-out target or motion demonstrating model reuse if that experiment succeeds.

The first statement requires real task evidence. The second requires more than smaller training loss or retrospective replay. The third should be merged into the second if the experiment is too small to sustain a separate analytical contribution. Two well-supported contributions are preferable to three overlapping claims.

A technically stronger framing is possible if the evidence shows that modeling the *executed* attachment response materially improves planning relative to assuming perfect PVA tracking. That would explain why an otherwise familiar rope-planning pipeline becomes a substantive aerial robotics problem. Treat this as a testable hypothesis; current implementation alone does not establish the result.

Robot arms also have tracking errors, so imperfect actuation itself is not unique to flight. The useful result would quantify how aircraft response and cable transients interact in this operating regime, and show that identifying this response changes the outcome. Merely moving an established method onto a UAV is a weaker claim than demonstrating and explaining that effect.

### 4.2 Claims to retain, qualify or omit

| Claim | Recommendation | Evidence required |
|---|---|---|
| Targeted aerial whipping system | Central | Repeated physical trajectories and fixed task metrics |
| Model refinement improves prediction | Central | Same unseen recordings, same inputs/initialization/masks |
| Model refinement improves task execution | Central hypothesis | Frozen plans tested prospectively with matched planning rules |
| Model is reusable | Valuable | New command/target with no additional model fitting |
| Both UAV and cable learning matter | Conditional | Controlled component-update comparisons |
| Both residual networks are necessary | Do not assume | Fair trained variants, not only switching networks off |
| Data-efficient | Quantify cautiously | Total recording duration, independent takes and learning curve; a superiority claim also needs a comparator |
| Physics-informed / hybrid predictor | Appropriate description | Exact mechanics, residual placement and limitations |
| Fully coupled UAV–cable dynamics | Inaccurate for selected model | Explicit force coupling would require a different implementation |
| Online adaptation or real-time MPPI | Inaccurate for this execution | Current updates and full-maneuver search occur offline |
| First robot whip / first real–sim–real rope planner | Omit | Direct precedents exist |
| New system-identification algorithm | Omit | Standard supporting method, as intended |
| High impact power / learned thrust limits | Omit | Current measurements and model do not establish them |
| Autonomous end-to-end data preparation | Unnecessary | Manual processing can be documented and reproducible |

## 5. Method presentation and mathematical soundness

### 5.1 The forward predictor

Use an explicit schematic state-space description:

```text
xD[k+1] = FD(xD[k], delayed PVA[k]; thetaD, phiD)
a_root[k] = geometry(xD[k], measured body-to-attachment transform)
xC[k+1] = FC(xC[k], a_root[k:k+1]; thetaC, phiC)
y_hat[k] = observed marker positions extracted from xC[k]
```

Here `a_root` denotes the attachment pose/boundary trajectory, not acceleration. In the final manuscript, choose a symbol that cannot be confused with the PVA acceleration column. `thetaD/thetaC` are effective nominal parameters; `phiD/phiC` are learned corrections. Draw a feedback arrow only for the onboard UAV tracking loop. The cable prediction has no reaction-force arrow returning to the aircraft model.

Explain that desired acceleration is kinematic acceleration and that commands are sampled packets. Perfectly integrated reference knots do not imply that the physical aircraft follows a continuous spline exactly. Delay, body-frame geometry and controller response affect the boundary motion that drives the cable. These details belong in the system model because they determine what is actually being learned and planned.[^29]

### 5.2 Identification objective and stages

The correct method name is **staged, regularized nonlinear system identification by simulation-error minimization**. For each stage, recursively simulate from a causal initial-state estimate, compare valid measured quantities, normalize within takes, and combine data families with fixed weights. Schematically:

```text
L_stage = sum_f alpha_f mean_takes_in_f(mean_valid_rollout_errors)
          + nominal parent regularization
          + residual magnitude and parent-output regularization.
```

The unnormalized family weights are new whip `1`, all prior training whips `0.5`, and preliminary training `0.5`; normalize over available families. All older whips share one family. This is not equal weighting of every generation, nor weighting by the number of frames in a long take. The robust error uses fixed engineering scales, which should not be called sensor-noise standard deviations.[^28]

Fit aircraft response and attitude, then its residual and attitude refinement; fit cable parameters under measured attachment motion, then its residual. Nominal positive parameters use bounded nonlinear least squares in log coordinates, with training-based delay profiling. Neural corrections use Adam with gradients through the rollout. Parent parameters and networks initialize the update; optimizer state is reset for the changed dataset. Freeze the training-selected, numerically checked candidate before operational validation.[^28]

**There is no joint command-to-tip loss in this implemented fitting procedure.** The complete cascade is evaluated afterward. This is a legitimate staged method, but better component fits need not yield better combined prediction. Altered attachment errors can remove a previous cancellation with cable errors. The paper should expose this possibility and report the actual complete error rather than replacing it with conditional cable error.

Longer fitting windows are not automatically superior. They increase the importance of recursive behavior but can make nonlinear optimization and state-initialization sensitivity more difficult. The current independent initialized windows do not impose cross-window continuity constraints, so they should not be called constrained multiple shooting. Cite the relevant identification literature and state the actual 2 s preliminary drone, 1 s preliminary cable and full planned-whip windows.[^23][^28]

### 5.3 Planner and objective

Present the optimizer as **MPPI-inspired offline sampling in trajectory-parameter space**. It evaluates candidate jerk sequences under the complete predictor, uses exponential score weights within proposal families, and selects commands using contact-first ordering. Deterministic incumbents and seed evaluations are additional to the 512 random samples. The 1.5 s horizon is the maneuver search interval; it is neither computation time nor a demonstrated receding-horizon feedback rate.[^21][^29]

The success rule and the preference score have different jobs. A geometric tip entry establishes modeled task success. The score favors the desired fold/cast behavior and fast forward contact while accounting for other penalties. The selected speed term is:

```text
contact_speed_bonus = 1600 * v_forward^2 / (4^2 + v_forward^2)
```

It is applied at feasible successful contact; there is no current hard 4 m/s success cutoff or earlier-hit bonus. Explain the exact forward-direction convention in the implementation. Publish the complete reward and ranking settings, not just this attractive term. A saturated bonus also means that doubling its weight need not produce a faster trajectory.

Disclose the seed command bank and the archived shape/tangent reference used as motion priors. Freeze shared development priors before the clean campaign; no seed may silently incorporate a later clean-campaign winner. Reroll every seed under the candidate model. If the study deliberately accumulates new seeds, give the unchanged-model comparison equivalent access and search. These are legitimate engineering choices, but they prevent a claim that the optimizer discovers whip structure with no prior motion information. Recovery is appended by a shared trajectory generator and checked separately; it is not an optimized cable-settling phase or a phase learned by PPO.

## 6. Existing development evidence

### 6.1 Three generations, two updates

The canonical flown lineage is `M0 → M1-full → M2-frozen-refit-v1`. The rejected gain-only `M1` is a sibling, and the earlier full M2 and its exact frozen-method refit are the same generation. Reproducing a fit is a useful numerical reproducibility check; it is not an independent adaptation experiment.[^28][^30]

Development M0 had its cable residual disabled; M1 introduced an active cable residual. Consequently the historical M0→M1 change includes model-capacity and pipeline-development changes, not only additional data under a fully fixed model class. This is another reason to keep the existing lineage separate from a fresh same-class paper experiment.[^30]

| Comparison | M0 | M1 | M2 | Valid interpretation |
|---|---:|---:|---:|---|
| Complete tip RMS on the same three M2 recordings, cm | 14.860 | 7.989 | 7.034 | Matched causal-history replay; none trained on these recordings |
| Drone RMS on those recordings, cm | 9.407 | 7.058 | 5.757 | Better mean aircraft prediction on this command family |
| Conditional tip RMS using measured attachment, cm | 10.857 | 7.336 | 6.345 | Cable diagnostic, not deployment prediction |
| Own original forecast tip RMS, cm | 17.17 | 8.77 | 9.38 | Different flown commands and historical planning choices |
| Observed 5 cm virtual entries | 0/5 | 0/5 | 0/3 | No demonstrated reliable hitting |

These numbers come from the checksummed project comparison, not published results.[^30] The matched complete-error reduction is 52.7% from M0 to M2 and 12.0% from M1 to M2, but the latter is a mean over only three takes of one command family. Take 002 worsens from 6.38 to 8.97 cm. Do not present the thirteen flights as independent model-update runs or the thousands of tracked frames as independent trials.

The matched diagnostic uses 0.4 s of causal aircraft history and 1 s of cable history. Its processing also standardizes timing, grids and masks; the difference from original-forecast error is not an initialization-only intervention. A future table should name the initialization and processing contract in its caption rather than letting “complete prediction” imply that all values were available before flight.

The own-forecast row is operationally meaningful: it records how well the model predicted the command actually chosen at that time. It does not isolate identification quality because both commands and some planning choices changed. The common-input comparison addresses a different question and should occupy a separate figure or clearly labeled panel.

Launch mismatch is a plausible contributor, not a complete explanation established by the data. Development diagnostics include drone start displacement up to 8.83 cm and estimated tip speed up to 0.136 m/s. A diagnostic replay initialized from measured prelaunch history can assess sensitivity; it cannot replace the original forecast or retroactively make the physical open-loop procedure more informed.[^29]

### 6.2 PPO's present role

The latest sustained-exploration PPO job has completed its configured 2,097,152-attempt review budget. Its frozen best checkpoint is from 2,080,768 attempts, with a modeled hit at about 1.0866 s and score 1934.77; the supervisor produced a complete recovery rehearsal. These are simulation development results, not real PPO flights or proof of convergence.[^31]

This makes PPO credible as an alternative trajectory generator in the current scenario. It does not make a clean MPPI–PPO superiority comparison: MPPI has motion seeds, PPO starts from fresh weights, action parameterizations differ, and training effort is separate from inference time. For this paper, retain PPO outside the central M0→M2 result unless a specific comparison can be completed without weakening the physical experiment.

## 7. Recommended experiment

### 7.1 Separate model accuracy from task performance

Use two linked studies. **Study A evaluates frozen models on the same unseen command recordings. Study B evaluates the real commands that each model plans.** This separation prevents an easier new trajectory, a changed reward, or favorable launch conditions from being mistaken for a better model.

| Study | What remains fixed | What changes | Main outcome |
|---|---|---|---|
| A: prediction | Recorded commands, causal histories, timing, masks, scoring interval | M0/M1/M2 predictor | Complete command-to-tip RMS and target-approach prediction error |
| B: deployment | Hardware/controller, preparation, target, objective, seed bank, search rule | Model used to plan commands | Real closest tip distance; observed entry rate; speed at entry |
| C: reuse, if feasible | Final model and all fitting choices | Target or command family not used for adaptation | Prediction and physical performance without new fitting |

For Study A, a final untouched command bank recorded after model freezing is strongest. Include at least one motion not generated from the same adaptation target if claiming broader reuse. Report which command produced each recording and which data entered each model's ancestry. A model trained on a take cannot contribute a held-out score for that take.

Label Study A as causal-history model evaluation when it uses measured prelaunch history. Evaluate nominal-start original forecasts separately for the deployed procedure. Both are useful, but their interpretation must remain distinct even if they share a command-to-tip RMS formula.

For Study B, execute the baseline M0 plan and adapted plan in interleaved or randomized blocks under comparable conditions. The chronological data-collection loop alone confounds model generation with battery, session, operator practice and hardware drift. After freezing the models, revisiting their frozen plans in a common physical evaluation is much more informative. This requires no additional adaptation algorithm.

An M0-versus-final-M2 task comparison is the minimum useful endpoint comparison. Including M1 reveals the shape of the improvement, but should not consume all available repetitions. A second target, planned without fitting its recordings, often adds more scientific value than a third adaptation round. This stays within the whipping task; an unrelated demonstration is unnecessary.

### 7.2 Baselines and ablations by priority

| Priority | Comparison | Question answered | Practical scope |
|---|---|---|---|
| Essential | Frozen M0 versus adapted full model | Does the loop improve prediction and execution? | Matched replay plus prospective physical plans |
| Essential when later rounds receive more search | Extra replanning with unchanged M0 | Could more optimization or favorable seeds explain the gain? | Same seed bank and comparable search allocation |
| Essential for a system mechanism claim | Assumed perfect attachment/PVA tracking versus learned UAV response | Does the flying actuator's execution error matter? | Start with common-command prediction; plan/flight comparison if claiming control benefit |
| High value | Same model class, nominal-only update versus full nominal+residual update | Do learned corrections add value beyond parameter fitting? | Fit both on identical training roles; compare unseen data |
| High value | Drone-only, cable-only and full updates | Where does useful adaptation occur? | Saved stage/intervention diagnostics first; controlled fits for strong causal claims |
| High value | New target without refitting | Is the model reusable? | One meaningful displacement within the verified envelope |
| Optional | Simple low-dimensional task-error correction | Is reusable modeling worth the effort for repeated strikes? | Matched trial budget if implemented |
| Optional | Another trajectory optimizer or PPO | Is the result planner-specific? | Same predictor and objective; disclose priors and total compute |
| Optional | Cold refit versus parent initialization; replay ablation | Why these identification choices? | Required only for claims about their superiority |

Do not confuse a **removal diagnostic** with a trained ablation. Switching off an adapted residual measures the immediate effect of removing it from a jointly compensated model. It does not estimate how well a separately fitted nominal-only model could perform. Likewise, cross-combining a drone model from one generation with a cable model from another can diagnose sensitivity, but is not automatically a fair competing fitted system.

The current nominal-only-update comparison should preserve the declared model class and clarify whether inherited residuals are frozen or reset. A completely neural-free model is a different baseline. These alternatives answer different questions and need explicit names in captions.

### 7.3 Trial budget and statistical unit

Choose the budget around the smallest scientifically meaningful improvement and available repeatability, not a generic flight count. The earlier 90-flight proposal is one possible factorial design, not a power analysis or an ICRA requirement. A compact study with repeated baseline/final-model comparisons and one transfer target can be stronger than a wide matrix with one or two flights per cell.

Use a take as the within-session experimental unit and retain session/battery-block identity. Repeated executions of one optimized command measure physical repeatability; independent optimizer seeds measure search variability; independent fresh adaptation campaigns measure pipeline variability. These are different sources of variation. If only one adaptation campaign is feasible, state that scope and do not portray it as population-level evidence that two updates always suffice.

Show every take, with means or medians and suitable intervals. Use paired differences for Study A. For Study B, display block/session structure and avoid treating frames as replicates. At small sample counts, emphasize the raw distribution and uncertainty rather than a fragile significance claim. Report failure and missing-outcome counts explicitly. Do not stop collection because the current mean looks favorable.

## 8. Measurement, initialization and validity

### 8.1 Primary metrics

Prespecify the main real metric as minimum 3D free-tip distance to target center during the common `[0, 1.5] s` task observation window. Report sphere entry separately. Because nearest approach ignores timing errors, also report complete tip RMS and the predicted-versus-measured approach timing. For entries, report directed speed at first entry; for misses, label speed at nearest approach as a different quantity.

More precisely, `[0, 1.5] s` is a common **task observation window**. Current commands can enter appended recovery after a modeled hit near 1.1 s, so this window includes early recovery in those cases. State this explicitly and retain the phase boundary in every plot. A primary fixed-window metric avoids model-dependent truncation; an additional pre-recovery metric can be reported as a separately defined diagnostic.

Track uncertainty matters at this scale. At 5 m/s, a 10 ms timing error corresponds to 5 cm of along-path displacement. This arithmetic illustrates sensitivity; it is not a measured clock error in the experiment. Quantify target-location, marker and synchronization uncertainty, and show how plausible timing shifts affect the conclusion. Do not independently shift each trajectory to minimize its reported error.

An interpolated path can suggest a between-frame crossing, but interpolation across a large missing-marker gap cannot establish a hit. Retain visibility masks and distinguish observed misses, observed entries and indeterminate outcomes. If physical contact is claimed, add independent target/video evidence and stop free-cable prediction/fitting at the first external contact. The trial still counts toward task performance.

### 8.2 Launch state

The practical open-loop sequence is **plan first, prepare the state afterward, then execute**. A long planning run does not require holding the originally measured cable state unchanged throughout computation. It requires a repeatable preparation state at dispatch. Ten seconds of hover is a useful operational starting point, but its duration alone is not evidence that the cable is stationary or that the aircraft is at the nominal origin.

Record current prelaunch position, orientation, cable configuration and motion. A manually reviewed readiness procedure is acceptable if it uses stated tolerances and is applied consistently. Rejected preparations and launch-state deviations should remain visible. Avoid adding an untested state-dependent trajectory transform just before clean collection.

For diagnosis, report two forecasts when useful: the original nominal-start forecast available before flight, and a separate causal measured-start replay. The second can reveal how much error comes from initialization, but it must not be labeled the forecast used by the planner. If future work uses launch-distribution optimization, freeze that method before the relevant evaluation; it is not part of the current results.

### 8.3 Manual data processing and model selection

Manual trimming and take pairing are compatible with sound research. Preserve raw recordings; record trim boundaries, pair identity, frame transform, marker map, clock estimate, visibility rules and exclusion reasons. Assign whole-take roles before fitting and prevent overlapping windows from crossing roles. The scientific requirement is reproducibility and protection against outcome-driven choices, not a fully automatic interface.

Use training data for fitting and checkpoint selection. Use operational validation for a declared adoption decision, understanding that it is then part of the experimental development loop. Keep a final evaluation set out of both roles. A test take inspected repeatedly while adjusting rewards, preprocessing or model architecture is development data even if its filename still says “validation.”

Freeze the same model class for fresh M0 and subsequent updates. Otherwise apparent adaptation may partly reflect adding residual capacity at a later generation. The existing protocol identifies a fresh-M0 driver consistency gap; that should be resolved before clean collection, without rewriting historical evidence. Changes to controller settings, hardware or the target definition create new conditions and must be recorded.[^29]

### 8.4 Numerical and physical limits

Report the selected integrator and representative time-step/discretization sensitivity. The existing selected-M2 8/16/32-substep check is useful: the modeled hit persists, but the 8-versus-32 tip difference reaches 2.72 cm at an instant. That is not negligible relative to a 5 cm target, and a single trajectory does not establish general numerical convergence.[^29]

Bounded learned corrections can still inject energy or compensate for erroneous parameters. The archived stationary-hover residual diagnostic remains relevant. Use long-enough stationary and full-maneuver checks to identify drift and report the model's domain. This does not require proving passivity, but it does rule out an unsupported claim of physically conserved learned dynamics.

## 9. Reviewer questions and direct answers

These are analytical expectations for this particular paper, not a quoted ICRA scoring rubric. The official reviewer guidance asks reviewers to describe the contribution and justify their assessment; it does not prescribe a fixed number of ablations or flights.[^27]

| Likely question | Best evidence or response |
|---|---|
| What is new beyond arm whipping and real–sim–real casting? | Explain aerial command-tracking uncertainty, retained free-tip task and prospective reusable-model evidence; cite the closest papers prominently |
| Is this merely MPPI applied to an existing simulator? | Show why measured UAV response and iterative refinement matter on hardware; implementation effort alone is insufficient |
| Why not correct the command directly across trials? | Demonstrate new-command/target reuse; discuss ILC/IRP honestly; add a simple correction comparator if making a comparative claim |
| Is the cable just a pendulum payload? | Show spatial deformation and marker timing; compare the task to both payload and distributed-cable precedents |
| Does a better fit actually produce a better strike? | Separate common-command model accuracy from prospective task performance |
| Did the reward change between generations? | Use one frozen objective/ranking/seed bank in the clean study; describe historical runs as development |
| Are residuals hiding a bad physical model? | Report nominal-only and residual diagnostics, retained-domain tests and the effective-parameter interpretation |
| Is the model mechanically coupled? | State the cascade and its limitations; show full-chain error rather than implying missing force coupling exists |
| Is success a tuned wave score? | Use fixed target geometry and observed tip entry; show whip style as a separate diagnostic |
| Why open loop when the start varies? | Specify preparation, measured launch distribution and repeatability; do not claim disturbance rejection |
| How much real data and compute does it take? | Report all calibration/update take durations, fit time, search time, retries and compute hardware |
| Are the results cherry-picked? | Show all takes, exclusions, original forecasts, ancestry and fixed reporting windows |

## 10. Eight-page manuscript and visual presentation

### 10.1 Suggested allocation

The current ICRA 2027 call permits eight pages **including references**.[^26] An approximate allocation is:

| Content | Pages | Purpose |
|---|---:|---|
| Abstract, introduction, physical-system teaser | 1.00 | State the task, difficulty and measured contribution |
| Focused related work | 0.70 | Compare closest arm, aerial and refinement work |
| System/problem formulation | 0.90 | Define commands, model cascade and task metric |
| Identification and refinement loop | 1.15 | Explain the repeatable update and data roles |
| Offline planner and execution | 0.65 | Explain search, priors, ranking and recovery |
| Experimental setup and comparisons | 0.65 | Make evidence interpretable |
| Results and analysis | 1.70 | Show task outcomes, matched prediction and key ablation |
| Limitations and conclusion | 0.35 | Bound the result precisely |
| References | 0.90 | Credit the most relevant primary work |
| **Total** | **8.00** | Adjust after typesetting |

Methods and results should dominate; repository history should not. The full implementation contract and supplementary artifact can carry detailed settings, but the paper must contain enough information to understand and assess the central claims without opening a repository.

### 10.2 Main figures

1. **Physical task and loop.** A real UAV/cable image, target and three or four time-stamped cable configurations, plus a compact real-flight → model-update → replan diagram. Distinguish the onboard tracking feedback from between-trial adaptation.
2. **Prospective task performance.** Per-take nearest distance by frozen planning model, 5 cm reference line, entry counts and session blocks. Include uncertainty and missed/indeterminate outcomes. Avoid a simulated-success bar beside a real-success bar without conspicuous labels.
3. **Matched model prediction.** M0/M1/M2 on the same recordings: complete tip error as the main panel, aircraft and measured-attachment cable errors as diagnostics. Use common axes and show individual paired changes.
4. **Mechanism and reuse.** One informative attachment-model/residual ablation and a held-out-target result. A marker-speed heat map or shape sequence can explain the whip, but should not replace task statistics.

For trajectory overlays, use identical coordinates and timestamps, preserve gaps, show the target to scale, and distinguish desired UAV commands, predicted UAV/cable motion and measured motion. Select the representative take by a declared rule such as median final-model task error; label a best-take teaser as such. A favorable overlay alone is not the experiment.

If showing a propagation diagnostic, plot arc length against time with marker speed or curvature as color, and identify the measured UAV reversal. This can document distal progression. A claim about energy flux would require a stronger mechanical analysis than the present geometry-based score.

### 10.3 Video

A short video should show the actual preparation and one full-speed flight before slow motion, then an original-forecast ghost, a baseline/adapted comparison and representative failures. Label model generation, target, playback speed and whether a scene is simulated or measured. Show the recovery as part of the operational sequence, without presenting it as learned cable settling.

The video should reinforce the paper's distributions rather than substitute for them. Avoid showing only the best M2 take and the worst M0 take. A synchronized pair chosen by a stated criterion is more credible and easier to interpret.

## 11. Suggested writing

### 11.1 Title options

**Recommended:** *Targeted Aerial Whipping through Iterative Model Refinement and Offline Trajectory Optimization*

Alternative, if emphasizing the implementation chain: *Model-Refined Aerial Whipping with a UAV-Carried Flexible Cable*

Alternative, if real target interception is demonstrated consistently: *Real-to-Sim-to-Real Planning for Targeted Aerial Whipping*

Avoid “high-power,” “real-time,” “robust,” “generalizable” or “beyond reach” in the title unless the final experiment explicitly supports the corresponding claim. Naming MPPI is optional; the system question is broader than the optimizer's acronym.

### 11.2 Abstract scaffold

The following is a writing template. Bracketed fields require completed clean evidence and must not be filled with development numbers without relabeling the study.

> A UAV can exploit the transient motion of a flexible cable to bring its free tip toward a target while keeping the aircraft away from the target region. Planning such maneuvers requires predicting both the aircraft's response to commanded motion and the resulting cable deformation. We present an aerial whipping system that combines an effective loaded-UAV response model with discrete cable mechanics and learned residual corrections. Recorded flights refine the same model through staged regularized simulation-error minimization, after which an offline sampling-based optimizer generates a new PVA command sequence. The maneuver executes without cable-state feedback, with onboard UAV tracking control and a prescribed preparation procedure. In [number] physical trials across [conditions], model refinement changes held-out command-to-tip error from [value] to [value] and real target-approach error from [value] to [value], with [entries/trials] observed target entries. [One supported ablation or transfer result.] These experiments characterize the benefits and limitations of model refinement for open-loop dynamic aerial manipulation.

If the final study does not demonstrate entries, write “target approach” throughout and explain the limitation. Do not retain a striking-success claim by enlarging the target after seeing the test data.

### 11.3 Introduction logic

Build four connected paragraphs. First, introduce the physical capability: a UAV can excite a retained flexible cable so its distal end approaches a stand-off target. Second, explain why nominal trajectory tracking is insufficient: the actual moving boundary depends on the loaded aircraft/controller response, while cable motion magnifies timing and initialization errors. Third, acknowledge arm whipping, real–sim–real casting and aerial flexible-cable control, then state the specific unanswered question addressed by this experiment. Fourth, describe the pipeline and its measured contributions, using numbers only after the final evidence is available.

Do not open with an unsupported industrial application or a broad assertion that dynamic cable manipulation is unexplored. A carefully posed laboratory capability can be a meaningful robotics contribution. Its importance should come from the physical challenge and experimentally useful insight.

### 11.4 Related-work structure

Use three compact paragraphs in the manuscript: **dynamic free-end manipulation and iterative improvement**; **aerial flexible cables and agile payload manipulation**; **hybrid prediction and real-data model refinement**. Give IRP, Wiggle and Go!, the aerial cable papers and Real2Sim2Real casting specific comparisons. Use broader foundational references to support model and optimization choices. The larger source inventory below is a research resource, not a requirement to crowd every citation into eight pages.

## 12. Submission timing and remaining decisions

The official ICRA 2027 call lists the paper deadline as **15 September 2026, 23:59 PST**, an eight-page total limit and double-anonymous review. It lists accompanying-video submission on 17–22 September after the current blackout, with a 180 s / 20 MB limit. The event is scheduled for 24–28 May 2027 in Seoul.[^26] The page contains some stale template fragments; use its explicit 2027 instructions and verify the portal's deadline display rather than silently interpreting its PST wording as another time zone.

As of 11 September, this is a short runway. A later video window is not permission to base the submitted paper on experiments that have not been completed. Write the methods and related work now, then decide whether the available physical evidence supports the intended claim by the submission deadline. If it does not, narrow the claim or choose a later opportunity; further reward tuning cannot substitute for evidence.

The practical priorities are: close the fresh-M0 model-class and execution/measurement gaps; freeze preparation and processing rules; run a focused matched baseline/adapted experiment; and demonstrate model reuse if feasible. Freeze the experimental budget and final analysis before collection. The more extensive ablation menu is optional and should follow the claims, not drive another redesign.

This report recommends paper framing and experiment priorities. It does not release the clean-flight protocol, change the selected model/CSV, start a new fit, or supersede the frozen historical data roles. The paper can use standard identification and still make a strong system contribution, provided the prospective physical evidence makes that contribution concrete.

## 13. Source scope

The assessment uses primary papers, author manuscripts/project pages for current acceptance status, official proceedings and the project's frozen method/evaluation records. Coverage is current to 11 September 2026, including several 2026 works particularly close to the proposed framing. Accepted papers with unverified proceedings details are labeled accordingly; preprints are not described as peer-reviewed publications.

This is a focused technical comparison rather than an exhaustive systematic review or proof of priority. Publication status can change, and papers outside the accessible/indexed set may be relevant. No cross-paper numerical superiority is inferred from incompatible experimental settings. Recommendations about experiments and contributions are this report's analytical judgments; the cited papers motivate them without prescribing this project's exact thresholds, reward weights or sample counts.

## Sources

[^1]: Cheng Chi, Benjamin Burchfiel, Eric Cousineau, Siyuan Feng and Shuran Song. *Iterative Residual Policy for Goal-Conditioned Dynamic Manipulation of Deformable Objects*. RSS 2022; expanded IJRR 43(4):389–404, 2024, first online 2023. [Conference paper](https://www.roboticsproceedings.org/rss18/p016.pdf), §§III–IV and Algorithm 1; [accessible manuscript](https://arxiv.org/html/2203.00663v2); [journal record](https://doi.org/10.1177/02783649231201201). Conference methods and journal metadata verified; direct expanded-journal PDF access was unavailable.

[^2]: Arthur Jakobsson, Abhinav Mahajan, Karthik Pullalarevu, Krishna Suresh, Yunchao Yao, Yuemin Mao, Bardienus Duisterhof, Shahram Najam Syed and Jeffrey Ichnowski. *Wiggle and Go! System Identification for Zero-Shot Dynamic Rope Manipulation*. arXiv:2604.22102, April 2026; [v2, 10 September 2026](https://arxiv.org/html/2604.22102v2), §§3–5. [Author project](https://wiggleandgo.github.io/) confirms acceptance to CoRL 2026; final proceedings metadata not verified.

[^3]: Yaolei Shen, Antonio Franchi and Chiara Gabellieri. *Aerial Robots Carrying Flexible Cables: Dynamic Shape Optimal Control via Spectral Method Model*. IEEE Transactions on Robotics 41:3162–3182, 2025. [DOI](https://doi.org/10.1109/TRO.2025.3562459); [full author manuscript](https://arxiv.org/html/2403.17565v2), §§II, V-E and VI. The 2024 arXiv date is not the final journal publication year.

[^4]: Chiara Gabellieri, Lars Teeuwen, Yaolei Shen and Antonio Franchi. *Manipulation of Elasto-Flexible Cables with Single or Multiple UAVs*. IROS 2025. [DOI](https://doi.org/10.1109/IROS60139.2025.11246978); [full author manuscript](https://arxiv.org/html/2503.04304v2), especially §V-A, Eq. (12), Tables I–II and §VI.

[^5]: Yifan Zhai, Elia Raimondi, Yunfan Ren, Ismail Geles, Yannick Armati, Jiaxu Xing and Davide Scaramuzza. *Learning to Throw: Agile and Accurate Cable-Suspended Payload Delivery with a Quadrotor*. [arXiv:2606.27603](https://arxiv.org/abs/2606.27603), 25 June 2026; [full paper](https://arxiv.org/html/2606.27603v1), §§III–V and Tables I, III and V. Preprint; accepted venue not established by the verified record.

[^6]: Vincent Lim, Huang Huang, Lawrence Yunliang Chen, Jonathan Wang, Jeffrey Ichnowski, Daniel Seita, Michael Laskey and Ken Goldberg. *Real2Sim2Real: Self-Supervised Learning of Physical Single-Step Dynamic Actions for Planar Robot Casting*. ICRA 2022, pp. 8282–8289. [DOI](https://doi.org/10.1109/ICRA46639.2022.9811651); [full manuscript](https://arxiv.org/html/2111.04814v2), §§IV–V; [institutional publication record](https://ieor.berkeley.edu/publication/real2sim2real-self-supervised-learning-of-physical-single-step-dynamic-actions-for-planar-robot-casting/). The arXiv landing page uses an earlier title.

[^7]: Jonathan Wang, Huang Huang, Vincent Lim, Harry Zhang, Jeffrey Ichnowski, Daniel Seita, Yunliang Chen and Ken Goldberg. *Self-Supervised Learning of Dynamic Planar Manipulation of Free-End Cables*. [arXiv:2405.09581](https://arxiv.org/abs/2405.09581), May 2024; [full v2](https://arxiv.org/html/2405.09581v2), §§IV–VI, especially §V-D. Final publication venue not verified; treated as a preprint.

[^8]: Krishna Suresh and Chris Atkeson. *Learning Dynamic Rope Manipulation Using Task-Level Iterative Learning Control*. RSS 2026, according to the [author project](https://flying-knots.github.io/). [Full manuscript](https://arxiv.org/html/2602.21302v2), May 2026 revision, §§IV–V. The demonstrated task is a flying knot, not point interception.

[^9]: Yi Yang, Xiang Fei, Lehong Wang, Chenhao Li, Zilin Dai, Henry Kou, Lu Li and Howie Choset. *DeformX: A Versatile Co-Simulation Framework for Deformable Linear Objects*. [arXiv:2606.22116](https://arxiv.org/abs/2606.22116), June 2026; [full paper](https://arxiv.org/html/2606.22116v1), §IV-C and Appendix VII-D/E. [Author project](https://deformx.github.io/) reports IROS 2026 oral acceptance; proceedings publication not independently verified. Quantitative reaching and qualitative horizontal striking are separate demonstrations.

[^10]: Aleksei Krotov, Marta Russo, Moses C. Nah, Neville Hogan and Dagmar Sternad. *Motor control beyond reach—how humans hit a target with a whip*. Royal Society Open Science 9:220581, 2022. [DOI](https://doi.org/10.1098/rsos.220581); [PMC record](https://pmc.ncbi.nlm.nih.gov/articles/PMC9533004/), §§2.6, 3.2 and Fig. 5. The published [article PDF mirror](https://upload.wikimedia.org/wikipedia/commons/f/fe/Motor_control_beyond_reach%E2%80%94how_humans_hit_a_target_with_a_whip.pdf) supplied full text when publisher access was intermittent.

[^11]: Mahdiar Edraki, Silvia Buscaglione, Rakshith Lokesh, John Peter Whitney, Alireza Ramezani and Dagmar Sternad. *Human-Inspired Robot Whip Manipulation: Preparatory Actions Increase Range and Reduce Control Effort*. ICRA 2025 Workshop on Robotic Manipulation of Deformable Objects. [Primary workshop paper](https://deformable-workshop.github.io/icra2025/spotlight/01_01_05_Edraki_Human.pdf), §II-D and results. This is a workshop paper, not a main-conference ICRA article.

[^12]: Antonio Rapuano, Yaolei Shen, Federico Califano, Chiara Gabellieri and Antonio Franchi. *Nonlinear Predictive Control of the Continuum and Hybrid Dynamics of a Suspended Deformable Cable for Aerial Pick and Place*. ICRA 2026 according to the [primary arXiv record](https://arxiv.org/abs/2602.17199); [full paper](https://arxiv.org/html/2602.17199v1), §§III–IV and Table III. Evaluation is numerical; final IEEE DOI was not verified.

[^13]: Simon Zimmermann, Roi Poranne and Stelian Coros. *Dynamic Manipulation of Deformable Objects With Implicit Integration*. IEEE Robotics and Automation Letters 6(2):4209–4216, 2021. [DOI](https://doi.org/10.1109/LRA.2021.3066969); [author manuscript](https://crl.ethz.ch/papers/DynamicManipulationRAL.pdf), §§VI–VII, manuscript pp. 6–8; [ETH record](https://www.research-collection.ethz.ch/handle/20.500.11850/479918).

[^14]: Yevgen Chebotar, Ankur Handa, Viktor Makoviychuk, Miles Macklin, Jan Issac, Nathan Ratliff and Dieter Fox. *Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience*. ICRA 2019, pp. 8973–8979. [DOI](https://doi.org/10.1109/ICRA.2019.8793789); [full paper](https://arxiv.org/html/1810.05687v4), §§III–IV, Appendix A and Table II. Commonly called SimOpt.

[^15]: Peide Huang, Xilun Zhang, Ziang Cao, Shiqi Liu, Mengdi Xu, Wenhao Ding, Jonathan Francis, Bingqing Chen and Ding Zhao. *What Went Wrong? Closing the Sim-to-Real Gap via Differentiable Causal Discovery*. CoRL 2023, PMLR 229:734–760. [Official proceedings](https://proceedings.mlr.press/v229/huang23c.html); [paper](https://proceedings.mlr.press/v229/huang23c/huang23c.pdf), §4.3, Table 1 and §5. Method name: COMPASS.

[^16]: Bohan Wu, Roberto Martín-Martín and Li Fei-Fei. *Rapid Adaptation of Particle Dynamics for Generalized Deformable Object Mobile Manipulation*. ICRA 2026 according to the [author arXiv record](https://arxiv.org/abs/2603.18246), March 2026; [full paper](https://arxiv.org/html/2603.18246v1), §§II–III and Table I. Method name: RAPiD; final IEEE DOI not verified.

[^17]: Shamil Mamedov, A. René Geist, Ruan Viljoen, Sebastian Trimpe and Jan Swevers. *Learning Deformable Linear Object Dynamics From a Single Trajectory*. IEEE Robotics and Automation Letters 10(7):7635–7642, 2025. [DOI](https://doi.org/10.1109/LRA.2025.3577421); [author university record](https://publications.rwth-aachen.de/record/1013888); [accessible 2024 manuscript](https://arxiv.org/html/2407.03476v1), §§4.3–5.3, Appendices A and D. Method details were checked in the preprint; final publisher text was not compared line by line.

[^18]: Yizhou Chen, Yiting Zhang, Zachary Brei, Tiancheng Zhang, Yuzhen Chen, Julie Wu and Ram Vasudevan. *Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects*. CoRL 2024; proceedings published 2025, PMLR 270:2996–3014. [Official proceedings](https://proceedings.mlr.press/v270/chen25d.html); [accessible full manuscript](https://arxiv.org/html/2406.05931v2), §§4–5 and Appendices A.3/B.4. Method name: DEFORM.

[^19]: Guillem Torrente, Elia Kaufmann, Philipp Föhn and Davide Scaramuzza. *Data-Driven MPC for Quadrotors*. IEEE Robotics and Automation Letters, 2021. [DOI](https://doi.org/10.1109/LRA.2021.3061307); [full author manuscript](https://arxiv.org/html/2102.05773v2), §§III-C–F and IV. Its control and observation boundary differs from this project's PVA-response predictor.

[^20]: Leonard Bauersfeld, Elia Kaufmann, Philipp Foehn, Sihao Sun and Davide Scaramuzza. *NeuroBEM: Hybrid Aerodynamic Quadrotor Model*. RSS 2021. [Primary proceedings paper](https://www.roboticsproceedings.org/rss17/p042.pdf); [accessible author manuscript](https://arxiv.org/pdf/2106.08015), model architecture in Fig. 2 and model-evaluation sections.

[^21]: Grady Williams, Andrew Aldrich and Evangelos A. Theodorou. *Model Predictive Path Integral Control using Covariance Variable Importance Sampling*. [arXiv:1509.01149](https://arxiv.org/abs/1509.01149), 2015; [full paper](https://arxiv.org/pdf/1509.01149), §§II–IV and Algorithm 1. This citation supplies the original sampling/control formulation; the selected project variant is explicitly described as MPPI-inspired offline search.

[^22]: Siyuan Luo, Bingyang Zhou, Chong Zhang, Xin Liu, Zhenhao Huang, Gang Yang, Zhengtao Han, Xiaotian Hu, Eric Yang, Rymon Yu, Ziqiu Zeng and Fan Shi. *FLASH: Fast Learning via GPU-Accelerated Simulation for High-Fidelity Deformable Manipulation in Minutes*. [arXiv:2604.17513](https://arxiv.org/abs/2604.17513), April 2026; [full paper](https://arxiv.org/html/2604.17513v1), §§IV–VI and Appendix IX-2. Preprint status; no real policy demonstrations does not mean absence of real material-calibration data.

[^23]: Antônio H. Ribeiro, Koen Tiels, Jack Umenberger, Thomas B. Schön and Luis A. Aguirre. *On the smoothness of nonlinear system identification*. Automatica 121:109158, 2020. [DOI](https://doi.org/10.1016/j.automatica.2020.109158); [author manuscript record](https://arxiv.org/abs/1905.00820). Relevant to simulation-horizon optimization and multiple-shooting terminology.

[^24]: Georgios Kamaras and Subramanian Ramamoorthy. *Distributional Treatment of Real2Sim2Real for Object-Centric Agent Adaptation in Vision-Driven DLO Manipulation*. IEEE Robotics and Automation Letters 10(8):8075–8082, 2025. [DOI](https://doi.org/10.1109/LRA.2025.3581744); [latest full author manuscript](https://arxiv.org/html/2502.18615v4), March 2026, Algorithm 1 and §§III–V; [author university confirmation of ICRA 2026 presentation](https://assistive-autonomy.ed.ac.uk/project/icra2026/). A [published correction](https://doi.org/10.1109/LRA.2026.3677515) exists; its text was unavailable, so no conclusion about its effect is drawn here.

[^25]: Miklós Bergou, Max Wardetzky, Stephen Robinson, Basile Audoly and Eitan Grinspun. *Discrete Elastic Rods*. ACM Transactions on Graphics / SIGGRAPH, 2008. [Primary paper](https://www.cs.columbia.edu/cg/pdfs/143-rods.pdf). Foundation for discrete rod mechanics, not validation of this project's particular damping or learned corrections.

[^26]: IEEE ICRA 2027 organizing committee. [Call for Technical Papers](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/). Accessed 11 September 2026; explicit 2027 dates, preparation rules and video section. The call also specifies disclosure requirements for AI-generated article content; consult those instructions when preparing the manuscript.

[^27]: IEEE Robotics and Automation Society. [Information for ICRA Reviewers](https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra/information-for-icra-reviewers/). Accessed 11 September 2026; reviewer responsibilities, scoring guidance and ICRA 2027 timeline.

[^28]: Project primary method record. [Frozen staged system identification, version 1](C:/Users/wts28/Documents/PHD/particle_filter_cable_project/docs/FROZEN_SYSTEM_IDENTIFICATION.md), 10 September 2026. Executable evidence and frozen contract: `runs/audits/M2-frozen-refit-v1`; implementation stages inspected in [whip_full_fit.py](C:/Users/wts28/Documents/PHD/particle_filter_cable_project/experimental_data/whip_full_fit.py), [whip_full_data.py](C:/Users/wts28/Documents/PHD/particle_filter_cable_project/experimental_data/whip_full_data.py) and related optimizers. Internal research evidence, not an external publication.

[^29]: Project primary audit and proposed protocol. [Aerial whipping: whole-system paper audit](C:/Users/wts28/Documents/PHD/particle_filter_cable_project/docs/PAPER_PIPELINE_AUDIT.md) and [Clean aerial-whip experiment: protocol release candidate 1](C:/Users/wts28/Documents/PHD/particle_filter_cable_project/docs/PAPER_EXPERIMENT_PROTOCOL.md), 11 September 2026. Selected model [record](C:/Users/wts28/Documents/PHD/particle_filter_cable_project/runs/adaptation/M2-frozen-refit-v1/candidate/model.json); offline optimizer [implementation](C:/Users/wts28/Documents/PHD/particle_filter_cable_project/planning/mppi_trajectory.py). The protocol is not yet released for clean collection.

[^30]: Project primary evaluation. [M0 → M1 → M2 system comparison](C:/Users/wts28/Documents/PHD/particle_filter_cable_project/docs/M0_M1_M2_SYSTEM_COMPARISON.md), 10 September 2026, especially model-class history and comparison definitions; [checksummed numerical report](C:/Users/wts28/Documents/PHD/particle_filter_cable_project/runs/evaluation/M0-M1-M2-system-review-20260910-v2/report.json) and [per-take tables](C:/Users/wts28/Documents/PHD/particle_filter_cable_project/runs/evaluation/M0-M1-M2-system-review-20260910-v2/TABLES.md). Current recordings are development evidence.

[^31]: Project PPO supervisor [completion status](C:/Users/wts28/Documents/PHD/particle_filter_cable_project/runs/audits/M2-ppo-persistent-20260911/status.json), read 11 September 2026; final best checkpoint, independent preview, recovery result and stop reason. This supersedes older still-running handoff notes for that job. No physical PPO validation is reported.
