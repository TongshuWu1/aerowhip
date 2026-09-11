# ICRA 2027 paper task: project and literature review

You are the dedicated paper-review and writing task for the UAV cable-whipping project. Build your understanding from the actual implementation, frozen experimental evidence and primary research papers. Complete an independent review before proposing the manuscript. Do not merely summarize an earlier assistant's conclusions.

## Workspace and organization

The current project and development evidence are in:

`C:/Users/wts28/Documents/PHD/particle_filter_cable_project`

This path is the authoritative source for the present review, including uncommitted documents and local runs. If your new task runs in an isolated Git worktree, its checkout may be older and may omit these artifacts. Read the current project using the absolute path above. Do not assume a default-branch checkout represents the current implementation. Do not copy the large data/run directories into your worktree.

Keep paper-review outputs in `paper_review/` within your task's workspace. Treat the authoritative project, raw recordings, fitted models, command CSVs, original predictions and selected flight packages as read-only for this task. Other development work may continue separately. Record the identities and timestamps of the evidence you inspect so later changes do not silently change your conclusions.

The requested deliverables are Markdown. Use precise code/file references, primary-source citations and clear distinctions between verified facts, interpretation, recommendations and unverified points.

## Research objective

The intended paper is about the complete system: a UAV excites a retained flexible cable to bring its free tip toward a target; preliminary real motions initialize a predictive model; offline planning generates a whip; recorded flights refine the model; replanning produces subsequent real maneuvers.

The contribution is not a newly invented adaptation algorithm. The central scientific question is whether a reusable model of loaded-UAV response and distributed cable motion, refined through a repeatable procedure, improves unseen-command prediction and subsequent physical target approach/interception.

Target ICRA 2027. MPPI is the recommended main planner; PPO is an optional comparison. Keep the paper focused on whipping. Swing-and-settle is retired and multi-target work is paused. Existing recordings are development evidence; the intended final paper experiment will be separate. Manual data processing is acceptable when its rules, transformations, exclusions and data roles are recorded consistently.

## Read first

Resolve the following relative paths against the authoritative project root above:

1. `AGENTS.md` and `HANDOFF.md`. They contain dated updates and historical statements. Check dates and actual status; do not treat every historical bullet as simultaneously current.
2. `docs/NEW_SYSTEM_CHECK.md`.
3. `docs/ICRA_2027_RELATED_WORK_AND_FRAMING.md` — the latest approximately 9,200-word literature and framing report, with 31 source records. Review its reasoning critically.
4. `runs/audits/icra2027-related-work-20260911/aerial_notes.md`, `whip_notes.md`, `sim2real_notes.md` and `report_check.json` — source details, publication-status qualifications and checks.
5. `docs/PAPER_PIPELINE_AUDIT.md`, `docs/PAPER_EXPERIMENT_PROTOCOL.md`, `docs/PAPER_WRITING_HANDOFF.md` and `docs/PAPER_READINESS_REVIEW.md`.
6. `docs/FROZEN_SYSTEM_IDENTIFICATION.md`, `docs/M0_M1_M2_SYSTEM_COMPARISON.md` and the exact artifacts they cite.

The experiment protocol is a release candidate, not proof that all release conditions have passed. Earlier proposed joint-fitting methods are not necessarily implemented. If documentation conflicts, trace the actual selected run's configuration, source snapshot and model ancestry before deciding what the paper can state.

## Review the implementation

Map the active first-party code and entry points. Inspect `simulator/`, `planning/`, `learning/`, `experimental_data/`, `fitting/`, `policies/`, `tools/`, `deployment/`, `config/`, relevant root application files, and relevant tests. Classify active, alternative, legacy and research-only paths. Do not confuse a dormant implementation with the one that produced the selected flight.

Trace the scientific pipeline end to end:

- Raw controller/OptiTrack pairing, command identity, units, frames, marker identities, attachment geometry, timing, trimming, visibility gaps, differentiation, causal initial-state estimation and whole-take roles.
- Preliminary M0 initialization and later model updates: the exact model classes, fit windows, losses, robust scales, regularization, family/take weighting, replay, parameter bounds, delay treatment, optimizers, gradients, stopping, checkpoint selection and post-selection validation.
- Desired PVA packets to effective loaded-UAV pose, rotated attachment, cable mechanics, residual corrections and predicted markers/tip. Review the equations, discretization, constraint treatment and what is actually coupled.
- MPPI proposal parameterization, seeds, prior motion reference, score, contact-first ranking, sample accounting, horizon, stopping, execution envelope, PVA export and appended recovery.
- Evaluation/replay/UI: original forecasts versus retrospective predictions, generation ancestry, fixed scoring windows, masks, task outcomes, units and aggregation. Verify that displayed claims correspond to the intended data/model.
- PPO's relationship to the same model/objective and its current evidence. Read completion records rather than launching training or relying on old running notes.

Inspect relevant existing tests and audit results. Use small isolated checks only when needed to resolve a specific uncertainty. Do not launch fitting, training, MPPI search, new physical trials or broad expensive reruns. Identify implementation gaps with concrete evidence and consequences; do not fix or redesign the production pipeline during this review.

## Facts and distinctions to independently verify

- Hardware: 145 g UAV, 17 g cable/marker assembly, 0.9525 m cable, 2S battery. Ten moving measured cable sites map to twelve simulated nodes.
- Selected planning start `[0,0,1.255]` m and target `[1.25,0,1.0]` m; fixed virtual target radius 0.05 m.
- Commands are 30 Hz desired tracked-origin PVA. Onboard aircraft tracking uses feedback; cable-task execution follows a frozen command sequence.
- The predictor is an effective loaded-UAV response cascaded into cable dynamics. It does not explicitly feed cable reaction forces back into UAV dynamics.
- The implemented estimator is staged, regularized nonlinear system identification by simulation-error minimization. Nominal stages use bounded nonlinear least squares; neural corrections use Adam through recursive rollouts. Combined command-to-tip error is evaluated after fitting, not jointly optimized.
- Current residual bounds are regularization, not learned thrust or acceleration limits. Effective fitted parameters need not be uniquely identified physical properties.
- Selected MPPI is an offline, MPPI-inspired whole-maneuver optimizer: 512 random samples/update, 1.5 s horizon, motion priors, contact-first selection and soft contact-speed preference. Do not claim exact MPPI guarantees or real-time cable-feedback MPC.
- Recovery is appended by a shared trajectory generator. It is not learned by PPO or a demonstrated active cable-settling phase.
- Canonical flown lineage: M0 → M1-full → M2-frozen-refit-v1. Rejected gain-only M1 is a sibling; original full M2 and its identical frozen refit are not separate generations.
- Development M0 had the cable residual disabled and M1 introduced it. Historical improvement includes model-capacity changes. The clean study must address same-model-class M0 construction.
- Matched M2-recording tip RMS of 14.86/7.99/7.03 cm is postflight causal-history evaluation with common processing, not the original nominal-start deployment forecasts. M2 regresses on one take.
- Original-forecast tip RMS is 17.17/8.77/9.38 cm on different flown commands. Current observed 5 cm virtual entries are 0/5, 0/5 and 0/3. Do not claim reliable real hitting or treat these development takes as a fresh blind test.
- A fixed `[0,1.5] s` task window can include early appended recovery after modeled contact near 1.1 s. Keep phase boundaries and scoring definitions explicit.

The selected physical MPPI rehearsal is `20260910-211435-608306-M2-frozen-refit-v1-whip`. Its package is `runs/flight_packages/20260910-211435-608306`; its model is `runs/adaptation/M2-frozen-refit-v1/candidate/model.json`. Verify exact hashes and associated forecasts from their manifests. Read the latest PPO status at `runs/audits/M2-ppo-persistent-20260911/status.json`; it had completed when this brief was written.

## Review the research independently

Follow the latest report's primary links and read the relevant methods and experiments. In particular, check IRP; Real2Sim2Real planar casting; Wang's planar free-end cable paper; Wiggle and Go!; task-level ILC; DeformX; Zimmermann's implicit-integration manipulation; Shen/Gabellieri/Rapuano aerial cable papers; Learning to Throw; SimOpt; Kamaras–Ramamoorthy distributional Real2Sim2Real; DEFORM; Mamedov's DLO identification; and the UAV residual-model foundations.

Preserve distinctions between accepted papers, published proceedings, preprints and workshop papers. Check full methods rather than search snippets. Some 2026 papers are very close, and some older preprints now have final journal citations. The report identifies access limitations and a published correction whose contents were unavailable; do not invent what inaccessible text says.

Assess novelty by capability and evidence, not a checklist of familiar components. Ground-robot identification plus open-loop rope striking already exists. Aerial flexible-cable identification and feedback control also exist. Imperfect actuation is not exclusive to UAVs. The strongest potential contribution is a carefully demonstrated aerial command-to-tip refinement/replanning system, with evidence explaining why its execution model matters and when the learned predictor can be reused.

Do not claim the first system without an adequate priority search. Do not compare published success percentages, RMS values or runtimes across incompatible setups. Do not label speed as impact power or geometric marker progression as measured energy flux. Literature motivates choices; it does not validate our exact reward, one-second history, target radius or residual bounds.

Verify the official ICRA 2027 requirements and deadline. The prior check found 15 September 2026 and eight pages including references; distinguish the paper deadline from the later video window. No result may be presented as completed before its evidence exists.

## Deliverables

Complete the review autonomously and produce:

1. `paper_review/PROJECT_AND_RESEARCH_REVIEW.md`: a source-grounded account of the actual pipeline, mathematical/implementation findings, closest-work comparisons, valid claims, remaining evidence gaps and prioritized recommendations. Include precise code references and distinguish implemented, tested, physically demonstrated and merely proposed behavior.
2. `paper_review/CLAIM_EVIDENCE_MATRIX.md`: each proposed contribution mapped to code/artifacts, current evidence, limitations, required experiment and relevant prior work. Separate deployment outcomes from causal-history model diagnostics.
3. `paper_review/MANUSCRIPT_PLAN.md`: a focused eight-page structure, title/contribution options, figure/table plan, essential versus optional baselines, and an abstract scaffold with unmistakable placeholders for future results.

Prioritize matched unseen-command prediction, prospective replan-and-fly performance, comparable frozen-M0 search, execution/launch validity and one meaningful model-reuse test. Do not demand a new adaptation algorithm, every possible ablation, a large arbitrary flight count or an unrelated task. The methods can be settled while the final evidence remains pending.

Finish with a concise recommendation about the paper's strongest defensible story and the smallest set of unresolved issues that materially affects it. Preserve disagreements with the earlier research report when supported by evidence. Do not stop after acknowledging this brief or proposing another review plan.
