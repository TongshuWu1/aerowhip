# Related work for adaptation after aerial whipping trials

Research notes, 5 September 2026. This is a focused literature review, not an exhaustive novelty assessment. Methods sections were inspected for IRP, DEFORM, Learning on the Fly and Wiggle and Go; other entries were screened using primary abstracts, institutional records or author project pages. No reported performance numbers are directly comparable to our current simulator metrics.

| Work | Verified relevance | Implication for this project |
|---|---|---|
| [IRP, RSS 2022](https://arxiv.org/html/2203.00663) | Predicts the effect of action changes on a previously observed trajectory. Demonstrates real rope whipping, using a three-parameter arm primitive and primarily planar tip observations. | A direct between-attempt action-adaptation baseline. Its residual is a trajectory-change predictor, not an additive physical-model residual. |
| [DEFORM, CoRL proceedings 2025](https://arxiv.org/html/2406.05931v3) | Differentiable DER, learned integration corrections, and momentum-aware inextensibility. Trains physical and network parameters using recursively predicted trajectories; reports 100-step training. | Physics plus a neural correction is existing work. Evaluate multi-step prediction and constraint preservation. Appendix distinguishes the Theseus LM solver from SGD used for training. |
| [Learning on the Fly, RA-L 2026](https://arxiv.org/html/2508.21065v2) | Alternates residual-acceleration learning and differentiable quadrotor-policy adaptation. Compares full and low-rank updates. Policy gradients use the analytical model while stopping gradients through the residual. | Closely overlaps the proposed joint model/policy adaptation. Its feedback flight controller is different from our frozen cable-strike sequence. Benchmark runtime locally rather than borrowing its seconds-scale result. |
| [GenDOM, ICRA 2024](https://arxiv.org/abs/2309.09051) | Conditions policy behavior on object parameters; identifies them from a real demonstration using differentiable simulation. Extends GenORM. | Offers adaptation without changing policy weights. Our current actor is not explicitly parameter-conditioned; this would need an architectural/training change. |
| [Wiggle and Go, April 2026 preprint](https://arxiv.org/html/2604.22102v1) | Infers rope parameters from a diagnostic motion, then optimizes actions with CMA-ES. Reuses identification for striking, lobbing and draping. | Reusable rope identification is already demonstrated. Its reported CPU action-search cost motivates a measured GPU/gradient adaptation comparison, not an assumed speed advantage. |
| [Planar Robot Casting, 2021/2022](https://arxiv.org/abs/2111.04814) | Fits simulator parameters with Differential Evolution from real cable trajectories, then learns from simulated and physical examples. | Direct Real2Sim2Real cable-manipulation precedent; support friction distinguishes its dynamics from aerial whipping. |
| [SimOpt, 2018/2019](https://arxiv.org/abs/1810.05687) | Alternates policy learning and adaptation of simulation-parameter distributions using real rollouts. | Compare calibrated uncertainty against a single best-fit model. |
| [AdaptSim, CoRL 2023](https://irom-lab.princeton.edu/AdaptSim/) | Learns simulator adaptation to improve task performance; adapted parameters need not reproduce true dynamics. | Distinguish task-specific effective models from a reusable calibrated physical simulator. |
| [COMPASS, CoRL 2023](https://proceedings.mlr.press/v229/huang23c.html) | Learns relationships between simulator parameters and trajectory discrepancy to focus parameter updates. | Supports diagnosing which parameter group explains error before changing all parameters together. |
| [Residual Physics, RA-L 2024](https://srl-ethz.github.io/website_residual_physics/) | Learns distributed residual forces for high-dimensional soft models from sparse motion-marker data. | Relevant to ten-marker supervision. A force residual is one alternative to a position/integration residual; neither should be selected without rollout tests. |
| [Distributional Real2Sim2Real for DLOs](https://www.research.ed.ac.uk/en/publications/a-distributional-treatment-of-real2sim2real-for-object-centric-ag/) | Uses likelihood-free inference to estimate parameter posteriors, then randomizes training using those posteriors. | A DLO-specific precedent for uncertainty-aware fitting and policy training. |
| [Aerial flexible-cable shape control, T-RO 2025](https://arxiv.org/abs/2403.17565) | Models coupled UAV/cable dynamics with a reduced spectral model and applies full-state NMPC in real experiments. | Important aerial-system reference. Full-state feedback and shape tracking are different objectives from a frozen impact maneuver. |
| [Aggressive quadrotor ILC, ICRA 2009](https://www.research-collection.ethz.ch/items/66771143-2261-4623-ade8-2a6aa2482eff) | Uses repeated flight experience to improve aggressive maneuvers with a simple model and iterative learning control. | Between-attempt feedforward correction is established; include a simple local correction baseline. |
| [Differentiable policy-gradient analysis, ICML 2022](https://proceedings.mlr.press/v162/suh22b.html) | Analyzes difficulties with first-order gradients in differentiable simulation. | Test gradients near contact/termination; retain strict event-based evaluation even if optimization uses smooth losses. |

## Proposed direction, inferred from the literature

Study adaptation between trials of an initially conditioned, open-loop aerial strike. Separate aircraft force-tracking error from cable dynamics, preserve a reusable model, and measure how quickly that model enables a new successful maneuver. This is a research hypothesis, not an established novel contribution.

Record successful and unsuccessful trials. Replay measured attachment motion to identify cable behavior separately from force-to-aircraft response, then validate the complete force-driven system. Fit pre-contact motion first unless target contact is explicitly modeled. Mix new trials with preliminary data and validate on whole held-out recordings.

Start with bounded physical-parameter adaptation. Add a small motion residual only when it improves held-out multi-step rollouts. Measure parameter sensitivity/ambiguity and maintain uncertainty; a good trajectory fit does not establish that each parameter equals its material ground truth.

For immediate command adaptation, optimize a small smooth correction around the actor's proposed force sequence, then validate and freeze it before execution. Thirty-Hz execution does not require ninety independent optimization variables: a lower-dimensional correction basis may be more practical with limited real data.

For later policy adaptation, compare (a) explicit dynamics-parameter conditioning with fixed weights, (b) differentiable full/low-rank weight updates, and (c) ordinary short PPO/SAC fine-tuning. Conditioning only covers variations represented during training; it does not automatically absorb arbitrary neural residuals. A useful intermediate step is distilling corrected force sequences across initial states, evaluated on unseen initial states.

## Minimum convincing comparisons

- Frozen initial model and actor.
- Direct action correction with the model frozen.
- Updated model plus identical-budget action optimization.
- Updated model plus policy adaptation.
- Ordinary RL fine-tuning with matched real-data and wall-time budgets.

Include a parameter-conditioned actor if the available development budget supports it. Distinguish literature implementations from project-specific adaptations of those baselines.

Report real trials to success, total adaptation wall time (data processing, fitting and command/policy optimization), held-out marker prediction versus horizon, strict impact success, pre-impact directed speed, drone travel and recovery. Speed is not a measurement of impact force; a force/impulse claim requires appropriate contact instrumentation or a validated contact model.

To support reusability, adapt using one subset of strikes and evaluate different target/initial-state distributions without refitting. New target locations establish within-task transfer; a separate maneuver is needed for a stronger cross-task reuse claim. Split by complete take/trial, preserve a final test set, and report variation across seeds and flights.

Read first: IRP, Learning on the Fly, DEFORM, GenDOM and Wiggle and Go.
