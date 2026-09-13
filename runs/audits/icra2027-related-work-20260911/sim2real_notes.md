# Primary-source notes: model refinement and deformable dynamics

Research checked 11 September 2026. Facts below are paraphrased from primary papers or their official proceedings/author institutional records. Each paper's fact record is deliberately short; comparison and recommendations are analytical judgments. No experiment or model was changed. Project grounding: `docs/FROZEN_SYSTEM_IDENTIFICATION.md` was read directly.

## Most consequential finding

Real-to-sim-to-real DLO manipulation already has an explicit, closely named published treatment: Kamaras and Ramamoorthy, RA-L 2025, presented at ICRA 2026. SimOpt additionally already adapts both actuator and rope properties. Therefore the paper cannot sell the generic loop, hybrid physics plus neural corrections, or modeling both robot and cable as new principles. The credible system distinction must rest on the UAV-actuated transient free-tip task, a reusable command-to-tip prediction chain, and prospectively demonstrated improvement after fixed-method model updates and replanning.

## Source records

### S1. SimOpt

**Metadata:** Yevgen Chebotar, Ankur Handa, Viktor Makoviychuk, Miles Macklin, Jan Issac, Nathan Ratliff, Dieter Fox. *Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience*. ICRA 2019, pp. 8973–8979; DOI `10.1109/ICRA.2019.8793789`. First arXiv October 2018; reviewed v4 March 2019.

**Facts:** SimOpt alternates simulated policy training, real policy executions, and updates to a distribution of simulator parameters by minimizing observed trajectory discrepancies. The real tasks include swing-peg-in-hole and drawer opening. The swing task's fitted distribution includes robot compliance, damping and action scaling, as well as rope bending/torsion properties, geometry and density. Its simulator evaluations run the policy in closed loop. Appendix A distinguishes this from replaying real actions open loop, especially when important state dimensions are unobserved.

**Primary:** [Full paper](https://arxiv.org/html/1810.05687v4), sections III-B–C, IV, appendix A, Table II; [author project](https://sites.google.com/view/simopt).

**Comparison:** Our retained point model, residual networks and offline command optimization are different engineering choices from learning a randomized simulator distribution and retraining an RL policy. The free-flying actuator and free-tip strike distinguish the task; simply having a rope and real/sim iterations does not. Avoid implying SimOpt only estimates the manipulated object.

### S2. Distributional Real2Sim2Real for DLO manipulation

**Metadata:** Georgios Kamaras, Subramanian Ramamoorthy. *Distributional Treatment of Real2Sim2Real for Object-Centric Agent Adaptation in Vision-Driven DLO Manipulation*. IEEE RA-L 10(8):8075–8082, August 2025; DOI `10.1109/LRA.2025.3581744`. Presented at ICRA 2026 according to the authors' university centre. Latest arXiv v4, 10 March 2026, has expanded title beginning *A Distributional Treatment...*.

**Facts:** The system infers posterior distributions over DLO parameters using likelihood-free inference/BayesSim, uses those distributions for simulated PPO training, then deploys the object-specific visuomotor policies without further physical fine-tuning. Its reaching objective brings the DLO body toward a 2D target above a table, using image keypoints and proprioception. Algorithm 1 uses a real rollout of an initial policy for inference. The paper explicitly distinguishes whole-body guidance from tip-specific rewards and leaves controller-stiffness calibration outside its scope.

**Primary:** [Latest full paper](https://arxiv.org/html/2502.18615v4), Algorithm 1, sections III-A–D and IV-A; [arXiv bibliographic record](https://arxiv.org/abs/2502.18615); [author university confirmation of ICRA presentation](https://assistive-autonomy.ed.ac.uk/project/icra2026/); [institutional accepted PDF](https://www.pure.ed.ac.uk/ws/portalfiles/portal/539352132/KamarasandRamamoorthyRA-L2025ADistributionalTreatment.pdf).

**Comparison:** This is mandatory related work for the proposed loop framing. Our distinction is retained-free-tip transient interception by a UAV, measured effective command response, and repeated model-update/replanning rounds. A point estimate can be valid, but its performance under launch variation needs empirical evidence; it has no posterior uncertainty guarantee.

**Bibliographic caution:** A correction exists: RA-L 11(5):6177 (2026), DOI `10.1109/LRA.2026.3677515`, listed in the author's ORCID and DBLP. Publisher correction content could not be retrieved. A secondary abstract describes three minor typographic changes only; do not repeat this as independently verified. Use latest v4 and flag correction in source notes, without suggesting invalid results.

### S3. DEFORM

**Metadata:** Yizhou Chen, Yiting Zhang, Zachary Brei, Tiancheng Zhang, Yuzhen Chen, Julie Wu, Ram Vasudevan. *Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects*. **CoRL 2024; proceedings published 2025**, PMLR 270:2996–3014. Cite the official proceedings instead of treating it as only a 2024 preprint.

**Facts:** DEFORM combines differentiable discrete elastic rods, neural corrections within integration, and an inextensibility procedure preserving momentum. It recursively predicts cable/rope trajectories and evaluates modeling, speed, tracking and manipulation. Its experiments use several DLOs with optical markers and industrial manipulators. The paper includes ablations of physical parameter tuning, residual learning and multistep training. The accessible manuscript uses one-second training sequences and explicitly optimizes multistep prediction loss.

**Primary:** [Official proceedings](https://proceedings.mlr.press/v270/chen25d.html), [full proceedings PDF](https://raw.githubusercontent.com/mlresearch/v270/main/assets/chen25d/chen25d.pdf); accessible [arXiv method](https://arxiv.org/html/2406.05931v2), sections 4.1–4.3, 5, appendix A.3 and B.4; [author project](https://roahmlab.github.io/DEFORM/).

**Comparison:** A DER-like simulator with residual learning and GPU execution is supporting machinery in our system, not a new generic model family. Our cable correction is a bounded acceleration discrepancy; do not describe it as implementing DEFORM's specific integration correction or momentum guarantees. Report numerical resolution and free-rollout error because a neural term can compensate for integration artifacts rather than material physics alone.

### S4. Single-trajectory DLO dynamics

**Metadata:** Shamil Mamedov, A. René Geist, Ruan Viljoen, Sebastian Trimpe, Jan Swevers. *Learning Deformable Linear Object Dynamics From a Single Trajectory*. **IEEE RA-L 10(7):7635–7642, July 2025**, DOI `10.1109/LRA.2025.3577421`; initial preprint 3 July 2024.

**Facts:** The model is a physics-informed neural ODE with a pseudo-rigid-body chain and learned nonlinear elastic interaction forces. The accessible 2024 manuscript trains each object from one roughly 30-second motion, using one-second rollouts, and evaluates different excitation trajectories. It optimizes training rollout initial states with model parameters, while test initialization uses preceding input/output history. It studies longer prediction horizons, discretization and data quality. Its preprocessing describes frame calibration, resampling, filtering and differentiation explicitly.

**Primary:** [Author university publication record](https://publications.rwth-aachen.de/record/1013888); [full accessible manuscript](https://arxiv.org/html/2407.03476v1), sections 4.3–4.4, 5.1–5.3, appendix A and D. **Method-detail qualification:** these detailed procedures were checked in the 2024 preprint; the final RA-L citation is independently verified, but final publisher text was not compared line by line.

**Comparison:** This supports rollout fitting and explicit causal test initialization, but does not establish that one second is universally optimal for our whip. Our near-full-whip fitting horizon should match the prospective maneuver timescale, while longer free-run evaluation tests accumulation. Manual preprocessing is scientifically acceptable when choices and transformations are specified. Do not import its 3.5 Hz filter into high-speed whip data without verifying signal bandwidth.

### S5. COMPASS

**Metadata:** Peide Huang, Xilun Zhang, Ziang Cao, Shiqi Liu, Mengdi Xu, Wenhao Ding, Jonathan Francis, Bingqing Chen, Ding Zhao. *What Went Wrong? Closing the Sim-to-Real Gap via Differentiable Causal Discovery*. CoRL 2023, PMLR 229:734–760.

**Facts:** COMPASS learns a differentiable mapping from simulator parameters to observed trajectory mismatch, organized by a learned causal graph. It uses this surrogate to identify parameter combinations and evaluates both trajectory alignment and downstream task performance. Its real air-hockey experiment uses a fixed real dataset for model tuning, retrains a policy in the revised simulator, and evaluates physical performance. The paper explicitly says fitting need not recover ground-truth parameter values; coupled parameters and local minima remain limitations.

**Primary:** [Official proceedings](https://proceedings.mlr.press/v229/huang23c.html), [PDF](https://proceedings.mlr.press/v229/huang23c/huang23c.pdf), section 4.3, Table 1, section 5.

**Comparison:** The relevant lesson is to distinguish improved predictive behavior from uniquely identified physical properties, and to assess the downstream controller after fitting. Our gains and delay describe effective loaded response. They do not by themselves identify maximum acceleration or motor thrust. We do not implement causal discovery and need not add it to justify a systems paper.

### S6. Nonlinear simulation-error optimization

**Metadata:** Antônio H. Ribeiro, Koen Tiels, Jack Umenberger, Thomas B. Schön, Luis A. Aguirre. *On the smoothness of nonlinear system identification*. Automatica 121:109158, November 2020; DOI `10.1016/j.automatica.2020.109158`.

**Facts:** The paper analyzes how optimization smoothness can deteriorate with simulation length for noncontractive models. It studies multiple shooting, splitting long simulations into shorter intervals with continuity constraints, and compares it with multistep prediction-error estimation.

**Primary:** [Author institutional record](https://research.tue.nl/en/publications/on-the-smoothness-of-nonlinear-system-identification/), [paper](https://arxiv.org/abs/1905.00820).

**Comparison:** Our fixed causal-initialized windows and whole-whip rollouts should be called **staged, regularized nonlinear system identification by simulation-error minimization**. Bounded trust-region reflective least squares and Adam are optimizers. This is not automatically multiple shooting: independent fixed initial states without jointly enforced continuity do not implement the paper's multiple-shooting formulation. A longer window is a design tradeoff, not inherently a better estimator.

### S7. MBPO: model use and exploitation

**Metadata:** Michael Janner, Justin Fu, Marvin Zhang, Sergey Levine. *When to Trust Your Model: Model-Based Policy Optimization*. NeurIPS 2019.

**Facts:** MBPO studies model-generated-data bias in policy optimization. Its practical algorithm uses short simulated rollouts branched from real states, motivated by analysis involving model generalization. It evaluates model-use choices rather than assuming that arbitrarily long synthetic rollouts are beneficial.

**Primary:** [Official paper record](https://proceedings.neurips.cc/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html), [paper PDF](https://papers.neurips.cc/paper/9416-when-to-trust-your-model-model-based-policy-optimization.pdf).

**Comparison:** We cannot import its policy-improvement guarantee into our offline MPPI-inspired optimizer. Our scientific test is whether the model remains accurate on newly optimized commands. Fitting loss on old trajectories cannot prove this, because the optimizer may seek poorly modeled states or amplify residual artifacts.

### S8. DEFT

**Metadata:** Yizhou Chen, Xiaoyue Wu, Yeheng Zong, Anran Li, Yuzhen Chen, Julie Wu, Bohao Zhang, Ram Vasudevan. *DEFT: Differentiable Branched Discrete Elastic Rods for Modeling Furcated DLOs in Real-Time*. arXiv:2502.15037, first submitted 20 February 2025. A final venue was not verified; label as preprint in this report.

**Facts:** DEFT extends differentiable elastic-rod modeling and residual learning to branched DLOs, including junction dynamics and grasping away from endpoints. It reports modeling and physical 3D shape matching/thread insertion experiments.

**Primary:** [Record](https://arxiv.org/abs/2502.15037), [full text](https://arxiv.org/html/2502.15037v1), sections VI–VII.

**Comparison:** Useful supporting context, lower priority than DEFORM for the single unbranched cable. Do not imply a new flexible-object model merely because ours learns a correction.

### S9. FLASH

**Metadata:** Siyuan Luo, Bingyang Zhou, Chong Zhang, Xin Liu, Zhenhao Huang, Gang Yang, Zhengtao Han, Xiaotian Hu, Eric Yang, Rymon Yu, Ziqiu Zeng, Fan Shi. *FLASH: Fast Learning via GPU-Accelerated Simulation for High-Fidelity Deformable Manipulation in Minutes*. arXiv:2604.17513v1, **19 April 2026**. Peer-reviewed venue not verified.

**Facts:** FLASH presents a GPU-native deformable simulation/learning system using a contact solver and parallel rendering/learning, with real towel and garment tasks. Its policy training uses simulated supervision and no real policy demonstrations. Crucially, appendix IX-2 nevertheless calibrates physical material parameters using a real corner-lift: it replays measured end-effector motion and searches Young's modulus/Poisson ratio to align simulated geometry with observed point clouds. Thus no demonstrations does not mean no real calibration data.

**Primary:** [Record](https://arxiv.org/abs/2604.17513), [full paper](https://arxiv.org/html/2604.17513v1), IV, V, VI, appendix IX-2.

**Comparison:** GPU acceleration and a fast integrated deformable manipulation stack already exist. Our compute results should be reported for our actual small-cable fitting/planning workload, with batch size/hardware/end-to-end timing, rather than claimed as broadly fastest. Abstract throughput numbers are incomparable to our complete fitting or optimization times.

### S10. RAPiD

**Metadata:** Bohan Wu, Roberto Martín-Martín, Li Fei-Fei. *Rapid Adaptation of Particle Dynamics for Generalized Deformable Object Mobile Manipulation*. arXiv:2603.18246, 18 March 2026. Author arXiv comment identifies ICRA 2026; final IEEE DOI not found during this check.

**Facts:** RAPiD first trains a policy with privileged deformable-dynamics embeddings in simulation, then trains modules to infer embeddings from recent depth images and robot observations/actions. Deployment freezes network weights and updates inferred embeddings online. Real tasks are object-end insertion and container covering with a mobile manipulator, evaluated on varied objects. This is latent context adaptation of a reactive policy, rather than refitting a reusable forward simulator between trials.

**Primary:** [Record](https://arxiv.org/abs/2603.18246), [full paper](https://arxiv.org/html/2603.18246v1), section II deployment and III experiments/Table I.

**Comparison:** Cite only if discussing online adaptation or future policy conditioning. Our current PPO has not demonstrated equivalent state/dynamics adaptation, and offline model updating should not be advertised as real-time adaptation.

## Research implications for this project

These are recommendations, not claims made by the cited papers.

### A precise formulation

Let M_k denote a frozen model generation; U a complete desired PVA sequence; z_0 the declared initial state. The planner uses the composition

`U -> predicted loaded-UAV motion -> rotated cable attachment -> predicted cable nodes/tip`.

Real execution produces a take containing commands, aircraft pose and cable markers. Reviewed training takes update M_k to M_(k+1) through the fixed staged simulation-error procedure. Replanning occurs with the same task objective/search contract. Independent takes assess two distinct propositions: prediction fidelity on common commands, and real task performance of newly optimized commands.

Do not replace that description with a single jointly minimized command-to-tip objective. The current code fits aircraft and measured-boundary cable components separately; the complete chain is an evaluation endpoint. Avoid calling the model fully bidirectionally coupled: cable reaction is absorbed, insofar as the data permit, into effective loaded-aircraft response, not fed back explicitly by the cable simulation.

### Essential evidence for the intended system claim

1. **Frozen-model cross-prediction on common unseen commands.** Compare M0, M1, M2 with identical logged inputs, causal initialization, timestamps, masks and scoring interval. Plot each take and aggregate paired differences; do not treat correlated frames as independent samples.
2. **Prospective replan-and-fly performance.** Freeze reward, target definition, search effort and launch procedure. Each generation's optimized CSV is evaluated physically before its outcomes can train a later model. Record all attempts, including failures and unobservable tip intervals. Model-generation labels are not independent sample units.
3. **Fixed-model replanning control.** Re-running the planner with unchanged M0 quantifies how much improvement comes from more search, reusable seeds or luck. It is especially valuable when comparing against repeatedly warm-started later generations. Use a matched planning budget and disclose seed origins.
4. **Model decomposition.** Show command-to-aircraft, measured-attachment-to-cable and complete command-to-tip errors. This explains whether improved individual stages compose, and can expose error cancellation. A single table can carry this without creating a separate modeling paper.
5. **Residual contribution tied to claims.** If the full hybrid model is central, compare fairly refitted nominal-only and full models on held-out prediction. Turning a trained residual off is a useful diagnostic but not a fair standalone physics-only baseline. Separate drone-only/cable-only removals are optional unless claiming both are necessary.
6. **Launch and timing audit.** Original nominal-start forecast measures deployed-system fidelity. A retrospective causal-history forecast diagnoses initialization error separately; it cannot replace the originally available prediction in prospective metrics. Clock uncertainty, observation gaps and differentiation choices must be recorded.
7. **One transfer axis.** To call the forward model reusable, a genuinely unused command or target is stronger than repeated copies of one CSV. A second held-out target is a focused option; many objects, tasks, or 90 flights are not automatic requirements. Scope conclusions to the hardware and target range actually tested.

### Optional evidence that depends on the claim

- Warm-start versus cold refit is needed to claim the adaptation algorithm itself saves samples or improves optimization; not necessary merely to use standard refitting in a systems contribution.
- Joint fitting versus staged fitting is relevant only if joint fitting is implemented or a superiority claim is made. Do not add it just for the paper narrative.
- PPO versus MPPI is optional. A fixed-scenario trained policy and a seeded trajectory optimizer answer different compute/data questions. If compared, include training/setup time, inference time, initial-state coverage, seed priors and task success separately.
- Ensembles, BayesSim/COMPASS replacements, uncertainty-aware planning and additional tasks could be future work. They are not prerequisites to a carefully bounded empirical paper.

### Claim language

Strong conditional system claim: **A calibrated forward-model and offline planning pipeline for UAV-actuated free-tip striking, whose repeated real-flight model updates improve unseen-command prediction and prospective striking performance.** The last clause requires clean evidence and cannot yet be inserted as an achieved numerical result.

Already defensible method statement: **We use staged, regularized simulation-error identification to update effective UAV response, cable mechanics and neural discrepancies from flight recordings, and reuse the resulting model for offline sampling-based trajectory optimization.**

Avoid: first real-to-sim-to-real DLO system; new adaptation algorithm; guaranteed monotonic improvement; learned physical actuator limits; exact material identification; fully coupled aerial cable dynamics; real-time MPPI; measured striking power inferred from tip speed alone.

## Search limitations

The closest identified loop paper and the listed final venue corrections should supersede older project documentation. This bounded cluster is not an exhaustive priority search. IEEE publisher access failed for some DOI pages; institutional records and full author manuscripts were used with version qualifications. No source's numerical result should be compared directly to our centimeters or runtime without matching the task, boundary condition, metric and hardware. Publication date is distinguished from preprint date where verified.
