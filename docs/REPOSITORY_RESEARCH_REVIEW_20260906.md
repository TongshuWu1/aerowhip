# Repository and research review — 6 September 2026

The project implements a substantial simulation and offline adaptation workflow for aerial cable whipping. Its strongest research direction is to show that separating aircraft-response error from cable-model error produces a reusable simulator that improves subsequent open-loop strikes with a small real-trial and computation budget. That benefit is still a hypothesis, not an established result.

## Scope and current checkout

Read `HANDOFF.md` first, followed by the architecture, research proposal, identification design, experiment protocol, adaptation guide, controller review, and related-work notes. Inventoried and syntax-parsed all **134 tracked Python files (25,051 lines)**, checked absolute imports into project packages for missing modules, and inspected the central physics, learning, initialization, adaptation, deployment, and UI paths. The accompanying `CODE_INVENTORY_20260906.md` records every tracked Python file. This is a repository-wide structural review with deeper inspection of the research-critical paths, not a line-by-line correctness proof of all source or CUDA kernels. The untracked NatNet SDK download is separate from the maintained project code.

No production code, active configuration, policy, recording, or historical report was changed. No production training or supervisor was launched. The protected recording and reserved policy evaluations were not inspected or evaluated. Tests used synthetic inputs and temporary artifacts.

Observed runtime: **Windows 11, Python 3.12.10, PyTorch 2.11.0+cu128, RTX 4080**. The handoff and `LAB_VALIDATION.json` describe an earlier RTX 5090 packaging workstation. This review does not establish Ubuntu or RTX 5080 compatibility, full training memory requirements, or flight readiness.

The active model's canonical SHA-256 matches the handoff:

`e759615bacd8c075d0fa6318f881d1a870ba6d00a0bdbbaf174660fca704c1c2`

However, this checkout lacks `runs/`, `results/`, `policy/`, the referenced constrained-calibration audit, and `data/baselines/20260905-181108-847814`. Consequently, the selected PPO checkpoint cannot be hash-verified or replayed here. The handoff's 84.375% development success and its prior comparison are historical reported evidence, not results reproduced in this review. The private experiment bundle is needed to reproduce that exact study; root configuration alone is insufficient.

## Research idea as implemented

1. Calibrate the attachment transform and cable model from preliminary motion capture, with complete-take training/validation roles.
2. Estimate the launch positions and velocities of a 12-node discretized cable, including the drone attachment.
3. Query a PPO actor along a private nominal simulation. Compile its actions into a finite total-world-thrust sequence and cutoff.
4. Execute that sequence once, with 20 Hz command holds, 100 Hz physics, and a fixed 20 ms follow-through bounded by the one-second horizon. Actual cable and hit measurements do not change the strike commands or cutoff. Low-level stabilization and subsequent recovery are separate.
5. Record the attempt. Replay measured attachment motion to assess cable prediction; separately replay commands to assess the coupled system.
6. Fit a bounded shared physical-model candidate, then locally optimize a force sequence. Validate the candidate before any baseline application or future execution.

The two updates must be described separately: shared dynamics parameters versus task-specific command correction. Current refinement leaves actor weights unchanged. The current active model has no neural residual. Despite the repository folder name, the active workflow is not a particle-filter estimator; the implementation centers on DDER simulation, causal state estimation, and PPO/SAC planning.

The plant represents drone translation at the shared cable root. It includes gravity, bending, internal damping, cable drag, and constraint reactions. It does not identify or simulate the full aircraft attitude/motor/controller response. Total modeled hanging mass is approximately 0.17509 kg and support force is approximately 1.71706 N. The 3.2 N force bound is provisional.

## Code map and methodological observations

| Area | Relevant source | Observation |
|---|---|---|
| Physics | `simulator/cable/dder.py`, `cuda_fixed_pcg.py`, `simulator/point_mass.py` | Substepped bending, implicit damping, mass-weighted length/velocity projection; separate differentiable and accelerated runtime paths require parity checks. |
| Observation, force, success | `learning/point_force_env.py` | 79-value observation; three normalized actions map to total world force. Strict success uses tip-first entry, 5 cm radius, directed speed at least 4 m/s, and angle at most 45 degrees. |
| Learning | `simple_ppo.py`, `simple_sac.py`, `deployment_rollout.py`, `sac_deployment.py` | PPO and SAC share deployment scoring. The independent execution/recovery outcome supplies whole-plan learning credit. SAC critics also receive the plan prefix. |
| Initialization | `experimental_data/state_initialization.py`, `flight_trials.py` | Both are causal, but the former uses a quadratic endpoint derivative and the latter a linear fit. Their noise/bias behavior differs. |
| Preliminary calibration | `recommended_fit.py`, `constrained_geometry.py`, `constrained_identification.py` | Recommended workflow fixes measured height, segment lengths, EI and Cb; fits lateral geometry and searches drag using training takes, then evaluates longer rollouts. |
| Flight-model update | `experimental_data/flight_adaptation.py` | Default drag-only TRF least squares uses bounded log parameters, robust residuals, trial balancing and autodiff JVPs. Separate boundary/coupled validation is implemented; baseline is not auto-applied. |
| Command correction | `learning/strike_adaptation.py` | Up to four XYZ correction knots, bounded by ±0.15 N per component, 12 Adam steps, fixed recorded cutoff, nominal strict rescoring. No next-launch or independent uncertainty acceptance yet. |
| Deployment | `deployment/planner.py`, `simulator/strike_plan.py`, `live_flight.py` | Finite offline plans and hash checking exist. `LiveFlight` is simulation. The received controller script is a reference, not an integrated learned-policy sender. |
| Desktop/provenance | `simulator/gui/main_window.py`, `workflow.py`, `learning/experiment_records.py` | Five-page workflow, candidate review, run snapshots, and per-evaluation records. |

Important distinctions for a paper:

- A measured-boundary replay is a conditional cable-prediction experiment; it has more information than command-driven flight prediction.
- Small predictive error does not establish unique material EI/Cb. Geometry, initialization, drag, actuation, and discretization can compensate for one another.
- A local command-gradient improvement in the simulator does not establish a useful real-world gradient.
- Refinement currently optimizes the final simulated tip state at the recorded cutoff. It does not search event times or optimize over measured uncertainty, as envisaged in the proposal.
- Rejected refinements still save a diagnostic `candidate_forces.csv` alongside a rejection status. Consumers must honor `result.json`; the CSV's existence does not indicate acceptance. The source trial is preserved.

## Findings requiring attention

### 1. Restore the exact experiment artifacts for reproducibility

The named selected checkpoint and calibration provenance directories are missing locally. `tools/export_lab_transfer.py` explicitly expects the selected run and checkpoint, so this checkout cannot currently produce that verified transfer. Obtain the preserved private bundle, verify hashes, and retain original run configurations. Do not reconstruct the historical experiment from today's defaults or restart training to replace it.

### 2. Planner validation does not check velocity compatibility

`deployment/planner.py:59` validates dimensions, finite values and segment lengths, but not the derivative of the length constraints. A synthetic hanging cable with only its tip given 1 m/s axial velocity was accepted unchanged. Its maximum segment-length rate was 1 m/s, incompatible with an inextensible cable. `PointForceWhipEnvironment.reset` also retains supplied velocities without projection.

The importer projects velocities, but external callers of the deployment NPZ interface can bypass that step. A future fix should reject incompatible velocities or explicitly project them while recording the correction. For each edge, check `(q[i+1]-q[i]) dot (v[i+1]-v[i])` with an appropriately defined tolerance. This finding was reproduced without flight data or policy weights.

### 3. Three maintained test fixtures are stale

The targeted run produced **55 passed, 3 failed** in 161.05 seconds:

- `tests/flight/test_attempt_execution.py:42` expects cutoffs `[1,20]`; the configured 20 ms follow-through correctly yields `[3,20]`.
- `tests/training/test_deployment_rollout.py:62` expects one physics step; the same follow-through produces three.
- `tests/training/test_deployment_rollout.py:153` allocates only two planning slots for a 0.2-second task whose current command period is 0.05 seconds. The mocked hit occurs on the third query, causing an indexing error.

These failures reflect outdated fixture timing assumptions; they are not evidence that production follow-through should be removed. Update the tests to state their timing contract explicitly and allocate from `env.control_step_count`. Tests and source were left unchanged in this review.

An additional **4 CUDA graph/eager parity cases passed** on the actual RTX 4080 in 12.95 seconds. Combined result: **59 passed, 3 failed**. This is targeted coverage, not a full-suite pass. GPU checks used small synthetic batches; they do not establish training capacity or physical-model accuracy.

### 4. Several research documents describe superseded state

`learning/README.md` still says lateral exploration is disabled and time cost is 1 point/s; active configuration uses all three action axes and 0.1 point/s. `PAPER_EXPERIMENT_PROTOCOL.md` contains historical fresh-start/empty-run statements. The proposal has a timing amendment, but its body and the older adaptation literature note still discuss 30 Hz. Treat the handoff and saved experiment configurations as authoritative; reconcile these descriptions before drafting a methods section.

### 5. The scientific bottleneck is causal attribution and transfer evidence

The architecture separates cable replay from coupled replay, but the latter still assumes instantaneous applied force. Recorded commands are not realized thrust measurements. Before concluding that cable parameters explain flight error, validate clock alignment, the rotating attachment transform, causal state estimates, and aircraft force response. Model acceptance also lacks a preliminary-data forgetting check and uncertainty-based full-strike acceptance. These are documented implementation gaps, not demonstrated adaptation results.

## Closest related papers

Primary abstracts and relevant method sections were checked online on 6 September 2026. This is a focused comparison, not an exhaustive novelty search. Each implication below is this review's inference; reported outcomes in these papers are not directly comparable to this project's success metric.

| Paper | Relevant result or method | Implication for this project |
|---|---|---|
| [Iterative Residual Policy — Chi et al., RSS 2022](https://arxiv.org/html/2203.00663v2) | Learns how changes in actions change a previously observed rope trajectory; refines actions between real attempts. Uses a three-parameter arm primitive for rope whipping. | The closest action-correction baseline. Its residual predicts trajectory changes; it is different from a physical motion residual. Show what explicitly updating your simulator adds. |
| [DEFORM — Chen et al., CoRL 2024; revised preprint 2025](https://arxiv.org/html/2406.05931v4) | Combines differentiable DER, learned integration correction, and momentum-aware inextensibility, with recursive prediction evaluation. | Differentiable rods plus a neural correction is established work. Evaluate your one-attachment coupled model, gradient quality, and long-horizon prediction; do not claim to reproduce DEFORM exactly. |
| [Learning on the Fly — Pan et al., RA-L 2026](https://arxiv.org/html/2508.21065v2) | Alternates residual dynamics learning with differentiable quadrotor policy adaptation. Uses analytical-model gradients without backpropagating through the learned residual in its main design. | Rapid model/policy adaptation is already demonstrated. Your current method instead corrects a frozen cable-strike sequence between trials and keeps actor weights fixed. |
| [Task-Level Iterative Learning Control — Suresh and Atkeson, May 2026 preprint](https://arxiv.org/html/2602.21302v2) | Uses a simplified robot/rope model and a quadratic program to turn critical task-state errors into command changes for a flying-knot task. | A strong low-complexity baseline for between-trial correction. Full trajectory identification must demonstrate an advantage over task-focused updates. |
| [Wiggle and Go — Jakobsson et al., April 2026 preprint](https://arxiv.org/html/2604.22102v1) | Infers rope parameters from diagnostic motion, then uses CMA-ES trajectory optimization. Reuses identification for striking, lobbing and draping. | Reusable rope identification is already prior art. Compare the benefit and measured cost of your coupled model and differentiable correction. |
| [Generalized-coordinate DER identification — Chen, Bretl and Pham, IROS 2025](https://arxiv.org/html/2310.00911v3) | Separates bending identification from twist/buckling identification using dedicated physical experiments. | Supports designing informative calibration experiments rather than jointly fitting weakly identifiable coefficients from arbitrary motions. |
| [Aerial Robots Carrying Flexible Cables — Shen, Franchi and Gabellieri, T-RO 2025](https://arxiv.org/abs/2403.17565) | Reduced spectral model and NMPC for coupled aerial cable position/shape control, including real experiments. | Relevant aerial-system modeling precedent. Its within-flight feedback and shape-tracking objective differ from a frozen impact maneuver. |
| [Learning to Throw — Zhai et al., June 2026 preprint](https://arxiv.org/abs/2606.27603) | Couples an analytical quadrotor model with rope/payload physics and learns agile targeted payload release. | A recent adjacent aerial manipulation comparison; targeted release is a different task, but an aerial robot plus flexible rope plus RL is not alone a novelty claim. |
| [Do Differentiable Simulators Give Better Policy Gradients? — Suh et al., ICML 2022](https://proceedings.mlr.press/v162/suh22b.html) | Analyzes how stiffness and discontinuities can undermine first-order gradient estimators. | Check finite-difference agreement and actual descent under strict resimulation, particularly near first contact and cutoff changes. |

Read IRP, Wiggle and Go, and Task-Level ILC first for the contribution/baselines; DEFORM and Learning on the Fly for implementation choices.

## A defensible experiment sequence

First establish repeatable sensing, causal initialization, and force-to-motion behavior. Then freeze the baseline actor and compare: unchanged model/actor; updated model with unchanged-actor replanning; fixed-model sequence optimization; updated-model sequence optimization; and a direct between-trial correction baseline. Keep real-trial budgets, initial information, success gates, and computation accounting comparable. Include direct per-state force optimization to assess whether PPO's initialization provides a useful speed or reliability advantage.

Report held-out boundary and coupled marker/tip error versus prediction horizon, strict hit and hit-plus-recovery rates, directed impact speed, drone travel, refusals/interventions, and total processing/fitting/planning time. Count the original prior search and policy training separately. Split whole trials and use independent repetitions; repeated deterministic nominal cases do not supply independent evidence.

Finally, fit on one target/initial-state family, freeze the model, and evaluate other families without refitting. That tests within-task transfer. A distinct maneuver is needed to support cross-task reuse. Add a neural residual only if physical-only held-out errors show a repeatable gap and the residual improves prediction and downstream outcomes without forgetting.

Candidate claim to test: **a small number of instrumented trials can improve a coupled aerial-cable model in a way that supports faster, more reliable new open-loop strikes, while preserving predictive usefulness beyond the strikes used for fitting.**

## Verification commands

```powershell
.venv/Scripts/python.exe -m pytest -q tests/physics/test_point_mass.py tests/physics/test_cable_core.py tests/physics/test_damping_precision.py tests/flight/test_deployment_package.py tests/flight/test_flight_adaptation.py tests/flight/test_attempt_execution.py tests/flight/test_strike_followthrough.py tests/calibration/test_state_initialization.py tests/calibration/test_differentiable_fit.py tests/training/test_deployment_rollout.py
.venv/Scripts/python.exe -m pytest -q tests/physics/test_cuda_graph_physics.py
```

All 134 tracked Python files parsed successfully. No unresolved absolute imports into the five checked local package namespaces (`simulator`, `learning`, `experimental_data`, `tools`, `deployment`) were found. These static checks do not validate external dependencies, runtime data paths, relative-import behavior, or all branches of execution.
