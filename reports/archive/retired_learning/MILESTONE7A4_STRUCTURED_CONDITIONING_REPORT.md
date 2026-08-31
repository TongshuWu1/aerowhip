# Milestone 7A.4 — Conditional State Utilization Diagnosis

Artifact root: `data/policy_training/diffusion_structured_conditioning_v1/2026-08-30T180111.928877Z`

## 1. Scientific problem

Milestone 7A.3 showed a healthy diffusion implementation and active context conditioning, but a large asymmetry between target and initial-state use. With matched noise, target changes produced action changes strongly correlated with the corresponding CEM change magnitude (`r = 0.965`), whereas state changes had essentially no such correlation (`r = -0.049`). The purpose of 7A.4 was to determine whether this came from preprocessing, weak state use, inadequate fixed-noise support, conditional capacity, or heterogeneous data before changing the architecture.

The decisive result is **FIXED_NOISE_SUPPORT_LIMITED**. A corrected, literally nested candidate audit raised the flat model's oracle success from 39.06% at 32 candidates to 64.06% at 128 candidates on the fixed 64-context diagnostic set. This +25.00 percentage-point gain satisfies the milestone's definition of a significant fixed-noise support limitation. Therefore the permanent structured-conditioning upgrade was not authorized and the selected generator remains the preserved 7A.3 flat model.

## 2. Why no new CEM was run

New CEM solves were exactly zero. All diagnostics used the already-paid-for partial-7A teachers, compatible Milestone-6A authoritative actions, the 7A.3 FULL checkpoint, and the 7A.3 fixed context/action manifests. Teacher generation, failure aggregation, scorer training, SAC, theta randomization, protected data, and hardware were not invoked.

## 3. Baseline 7A.3 result

`BASELINE_FLAT_CONDITIONING` is the immutable 7A.3 FULL EMA checkpoint. Its full-set results remain:

| Metric | Flat baseline |
|---|---:|
| Training oracle best-of-32 | 18.32% |
| Training contexts with a feasible candidate | 100.00% |
| External-development first candidate | 14.06% |
| External-development oracle best-of-32 | 60.94% |
| Correct external context oracle | 60.94% |
| Fully shuffled external context oracle | 25.00% |
| State-pair correct oracle | 40.63% |
| State-pair cross-swapped oracle | 12.50% |
| Target-pair correct oracle | 3.13% |
| Target-pair cross-swapped oracle | 0.00% |
| Clamp fraction | 0.05% |
| Median / p95 generator latency | 26.31 / 29.33 ms |

The complete baseline checkpoint, EMA weights, fixed noise bank, and manifests were not overwritten.

## 4. Actual existing conditioning implementation

The repository path actually executed in 7A.3 is:

1. The production context builder creates one root-centered, yaw-aligned, gravity-preserving 83-vector. Quaternion sign is canonicalized.
2. `FixedContextNormalizer` applies `(context - fixed_mean) / fixed_std`. It does not transform the action, and its statistics are not refit in 7A.4.
3. The flat condition encoder is `Linear(83,256) -> SiLU -> Linear(256,256)`.
4. The 49-D noisy action is projected by `Linear(49,256)`.
5. The integer diffusion timestep receives the 64-D sinusoidal embedding and `Linear(64,256) -> SiLU -> Linear(256,256)`.
6. Action, context, and timestep embeddings are added exactly once.
7. Four residual blocks then apply `LayerNorm -> Linear(256,512) -> SiLU -> Linear(512,256)` with residual addition. These blocks receive no repeated context modulation.
8. `LayerNorm -> Linear(256,49)` predicts epsilon.
9. Inference uses the repaired 25-step deterministic DDIM timetable from `t=95` through `t=0`, `eta=0`, and final-only `[-1,1]` clamp.
10. The fixed 32-vector noise bank makes a query deterministic.
11. The selected normalized 49-vector is decoded by the production decoder into 16 acceleration knots and `T_maneuver` in `[0.45,1.80]` s.
12. Command execution is ACTIVE, then the fixed 0.30-s analytic smooth SETTLE, then HOLD through `T_evaluation=2.40` s.
13. Physics uses cyclic padding to the fixed numerical shape 2048, the frozen UAV/residual/DDER model, and the unchanged hard scientific gates.

No masking, feature indexing, or context-normalization defect was found in that path.

## 5. Context feature statistics

Statistics were computed after the fixed production normalizer on the 202 TRAIN contexts.

| Block | Dims | Mean | Std | Effective variance/dim | Participation dimension | Near-constant dims | Max abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| UAV state | 10 | 0.0206 | 0.8484 | 0.6756 | 9.26 | 0 | 5.148 |
| Cable positions | 30 | -0.2193 | 0.9229 | 0.7564 | 25.59 | 0 | 4.629 |
| Cable velocities | 30 | -0.1509 | 0.8387 | 0.6704 | 28.22 | 0 | 4.559 |
| Goal | 6 | -0.0133 | 1.1232 | 1.2324 | 4.73 | 1 | 2.945 |
| Theta | 7 | approximately 0 | approximately 0 | 0 | 0 | 7 | 1.4e-6 |

The state blocks are neither zeroed nor numerically compressed: each has broad variance and almost all dimensions participate. The one constant goal coordinate is the expected horizontal direction z component. All theta features are constant because 7A.4 is nominal-theta only; the theta input is intentionally retained for later work. No normalizer refit is justified.

## 6. Denoising error by state/target strata

The saved flat model was tested across all repaired DDIM timesteps and state/target tertiles. At `t=95`, epsilon RMSE was 0.0808 / 0.0860 / 0.0893 for low/medium/high state distance and 0.0812 / 0.0865 / 0.0883 for low/medium/high target distance. Across the complete breakdown, state-distance/error correlation was `-0.024` and target-distance/error correlation was `-0.316`; the recorded dominant axis was target variation.

This does not explain the physical state weakness. Epsilon error is not monotonic in state distance and is a poor proxy for phase-sensitive hard success. The evidence therefore rejects a simple “high-state rows are numerically impossible to denoise” explanation.

## 7. Physical oracle by context group

The flat 32-candidate physical breakdown tracks context complexity:

| Group | Contexts | Oracle success | Any feasible candidate |
|---|---:|---:|---:|
| Target-only | 64 | 60.94% | 100% |
| State-only | 57 | 36.84% | 100% |
| Joint state+target | 58 | 12.07% | 100% |
| Edge | 56 | 14.29% | 100% |
| Partial-7A | 31 | 3.23% | 100% |

Failure clearly grows with heterogeneous state/task complexity, but it is not a feasibility-support collapse: every context contains a feasible generated action.

## 8. Block-ablation result

The same 64-context subset (16 per 6A group) was evaluated against the original physical tasks. Flat full-context oracle was 32.81%.

| Replaced with TRAIN mean | Oracle | Change from correct |
|---|---:|---:|
| UAV state | 10.94% | -21.88 pp |
| Cable positions | 4.69% | -28.13 pp |
| Cable velocities | 6.25% | -26.56 pp |
| Complete cable state | 14.06% | -18.75 pp |
| Goal | 20.31% | -12.50 pp |
| Theta | 35.94% | +3.13 pp |

The flat model behaviorally depends on all three state components. In particular, separately ablating cable positions or cable velocities is more damaging than ablating the goal. The weakness is therefore not “the flat encoder ignores state entirely”; it is inaccurate or incomplete conditional use.

## 9. State-only and goal-only shuffle result

With identical noise and evaluation against the original task, full-context oracle was 32.81%, state-only shuffle was 21.88% (-10.94 pp), and goal-only shuffle was 23.44% (-9.38 pp). Both blocks influence physical behavior. This again rejects a completely unconditional generator.

## 10. Matched-noise sensitivity result

| Change | Generated median L2 | Teacher median L2 | Median ratio | Magnitude correlation | Median direction cosine |
|---|---:|---:|---:|---:|---:|
| Target, same state | 0.3634 | 0.4638 | 0.824 | 0.966 | 0.622 |
| State, same target | 0.1647 | 0.4364 | 0.385 | -0.054 | 0.468 |

The model changes action when state changes, but its response is too small and does not scale with teacher variation. Teacher multimodality prevents treating exact action delta as a hard target, but the magnitude asymmetry is strong evidence of weak/inaccurate state adaptation.

## 11. Input/Jacobian sensitivity result

Central finite differences were propagated through the complete repaired 25-step sampler. Relative sensitivity fractions were UAV 20.93%, cable positions 22.33%, cable velocities 20.34%, goal 15.71%, and theta 20.68%. The nominal-only theta derivative is a counterfactual network sensitivity, not learned physics-conditioning evidence.

These derivatives show that all paths are connected, but they do not predict whether the induced action change is physically appropriate. Physical ablations and cross-swaps remain the primary evidence.

## 12. Corrected 32/64/128 candidate support result

The first implementation nested noise vectors but regenerated them in neural batches of 32, 64, and 128. Small GPU batch-shape rounding changes produced non-monotonic hard-success counts in this phase-sensitive system. That diagnostic implementation was corrected before final classification: the production 32 candidates are generated once, a second disjoint 32 are appended, and a disjoint 64 are appended. The resulting action sets are literal supersets, so an existing success cannot disappear.

Corrected flat results:

| Candidates | Overall oracle (64 contexts) | TRAIN subset | External-dev subset |
|---:|---:|---:|---:|
| 32 | 39.06% | 15.63% | 62.50% |
| 64 | 56.25% | 25.00% | 87.50% |
| 128 | 64.06% | 37.50% | 90.63% |

The 32-to-128 gain is +25.00 percentage points and the 32-to-64 gain is already +17.19 points. Fixed N=32 is leaving substantial learned support unused. This is a diagnostic conclusion only; production `N=32` was not changed.

## 13. Root-cause classification

Final Stage-A classification: **FIXED_NOISE_SUPPORT_LIMITED**, confidence HIGH.

Supporting evidence:

- no context construction, normalization, masking, or indexing bug;
- state conditioning is active but its matched teacher-response magnitude is weak;
- physical difficulty rises sharply from target-only to joint/edge groups;
- most decisively, a literal nested bank increases oracle success by 25 points from 32 to 128.

The limitation is mixed in a descriptive sense—state response remains inaccurate and the data distribution is heterogeneous—but the milestone's architecture gate requires fixed-noise support not to be dominant. That gate does not pass.

## 14. Why the architecture was retained

The corrected Stage-A result does not authorize the permanent structured conditioner. The selected generator therefore remains `BASELINE_FLAT_CONDITIONING`; its checkpoint and online contract are unchanged.

For transparency, a structured FiLM experiment had already completed after the earlier, insufficiently nested candidate audit incorrectly reported an 18.75-point gain and marked support as non-dominant. That experiment is preserved in the artifact root, is not promoted, and independently failed all requested method gates. This sequence is documented rather than hidden.

## 15. Non-selected structured experiment

The unpromoted experiment used the unchanged external 83-D input and internally encoded:

- UAV 10-D through a dedicated 10-64-64 MLP;
- each ordered cable node as joint `[position, velocity]` through a shared 6-64-64 encoder, learned node identity, and two kernel-3 chain convolutions;
- cable summaries from mean, max, c1, and c10 features;
- goal through a dedicated 6-64-64 MLP;
- theta through a dedicated 7-32-32 MLP;
- four residual denoiser blocks with condition+timestep FiLM scale/shift at every block.

The baseline has 1,249,329 trainable parameters; the structured experiment has 1,918,289 (1.535x), remaining in the same broad scale. Diffusion objective, direct 49-D action, scheduler, EMA, noise bank, physics, and data splits were unchanged.

It trained from scratch for 7,500 updates, selected the EMA checkpoint at update 2,500, and reached best internal validation epsilon loss 0.05774. Its generated distribution was numerically healthy: clamp fraction 0%, coordinate standard deviation 0.2897, p95 absolute coordinate 0.6187, range `[-0.802, 0.790]`, and duration range 0.838–1.329 s.

However, physical performance did not improve:

| Metric | Flat | Structured experiment |
|---|---:|---:|
| Training oracle best-of-32 | 18.32% | 18.81% |
| Training any feasible | 100% | 100% |
| External-dev first candidate | 14.06% | 18.75% |
| External-dev oracle | 60.94% | 54.69% |
| State correct | 40.63% | 43.75% |
| State cross-swapped | 12.50% | 28.13% |
| State correct-minus-swap gap | 28.13 pp | 15.63 pp |
| Target correct | 3.13% | 3.13% |
| Target cross-swapped | 0.00% | 3.13% |
| Clamp fraction | 0.05% | 0.00% |
| Median latency | 26.31 ms | 52.36 ms |

The experiment misses the training gate (at least 35% or +15 points), misses external non-regression by one of 64 contexts (54.69% versus 55% floor), weakens state discrimination, and loses target-pair separation. It is classified `CONDITIONAL_MODEL_STILL_WEAK` and not selected.

Its cable ablations are particularly diagnostic: replacing structured cable positions yields 32.81% versus 31.25% correct, replacing cable velocities yields 29.69%, and replacing the complete cable state yields 39.06%. Its absolute finite-difference cable sensitivities also fall from 0.1767/0.1609 in flat to 0.0313/0.0316. Thus this particular pooled chain encoder suppresses, rather than strengthens, useful distributed cable conditioning.

## 16. Candidate-count diagnostic for the unpromoted experiment

With the same literal nesting, structured diagnostic oracle is 35.94% / 51.56% / 60.94% at 32 / 64 / 128 candidates. Its TRAIN subset is 25.00% / 34.38% / 40.63%, and external subset 46.88% / 68.75% / 81.25%. It exhibits the same support-width effect and does not resolve it.

## 17. Inference latency

The selected flat model remains 26.31 ms median, 28.22 ms p90, 29.33 ms p95, and 63.27 ms maximum over 1,000 warmed 32-candidate queries. Context normalization and 25 DDIM steps are included; simulator, CEM, and scorer are excluded.

The unpromoted structured experiment measured 52.36 ms median, 55.33 ms p90, 56.74 ms p95, and 65.42 ms maximum. It meets the p95 <100 ms engineering goal but misses the median <50 ms goal and offers no compensating physical gain.

## 18. Current complete pipeline

The current selected neural pipeline is unchanged from 7A.3:

`measured UAV/cable state + target/direction + nominal theta`

→ construct the fixed 83-D local PolicyContext

→ apply the frozen context normalizer

→ repeat one normalized context across the frozen 32x49 Gaussian noise bank

→ run the 1.249M-parameter flat conditional epsilon network through the repaired deterministic DDIM schedule `[95 ... 0]` in 25 steps

→ clamp only the final 49-D normalized actions

→ obtain 32 candidate maneuvers.

The generator alone does not select an action in this milestone; all reported success is an offline physical oracle used to diagnose candidate support. Scorer development remains frozen. When a future scorer is re-enabled, it must rank these neural candidates without CEM or simulator calls online.

For offline scientific evaluation, every candidate passes through the single production 49-D decoder, then ACTIVE commands for decoded `T_maneuver`, the analytic 0.30-s smooth SETTLE, and stationary HOLD until 2.40 s. The complete frozen production physics is evaluated under the fixed-2048 numerical contract and unchanged scientific gates. No quarantine, simplified physics, legacy 1.20-s command semantics, or alternative CEM codec is used.

## 19. Whether more CEM is justified

**NO.** The present evidence does not support buying more labels. The selected generator already contains materially more successful support than the fixed 32 bank exposes, while the attempted structured conditioner did not improve training-manifold fit. More CEM would conflate a candidate-support/inference-design question with data coverage.

## 20. Recommended next step

Do not change production `N=32` automatically. The next reviewed task should isolate why the frozen 32 noise vectors cover the conditional modes poorly and determine whether a better deterministic fixed bank or candidate-generation coverage strategy can expose existing learned modes within the latency budget. That study should use only existing models/data first, preserve literal nesting, and keep scorer training paused until generator support at the intended online candidate budget is understood.

## 21. Integrity and prohibitions

- New CEM solves: **0**
- Scorer: **NOT TRAINED**
- SAC: **NOT USED**
- Production model/action/command contract: **UNCHANGED**
- Final 7A TEST: **NOT EVALUATED**
- Protected `fig8vertical_002`: **NOT EVALUATED**
- Real hardware: **NOT EXECUTED**
- Regression suite: **131 passed, 0 failed**

## 22. Final summary

    New CEM solves:
        0

    Baseline generator:
        7A.3 FULL

    Baseline training oracle:
        18.32%

    Baseline external-dev oracle:
        60.94%

    Root cause:
        FIXED_NOISE_SUPPORT_LIMITED
        with secondary state-response weakness and data heterogeneity

    Architecture change justified:
        NO

    Architecture changed:
        NO — selected generator remains immutable flat baseline
        structured experiment preserved but NOT PROMOTED

    Final generator:
        FLAT CONDITIONING

    Final training oracle:
        18.32%

    Final external-dev oracle:
        60.94%

    Correct-state oracle:
        40.63%

    State-swapped oracle:
        12.50%

    Correct-target oracle:
        3.13%

    Target-swapped oracle:
        0.00%

    32-candidate oracle:
        39.06%

    64-candidate oracle:
        56.25%

    128-candidate oracle:
        64.06%

    State conditioning:
        UNCHANGED — active but weak/inaccurate

    Target conditioning:
        PRESERVED

    Median latency:
        26.31 ms

    P95 latency:
        29.33 ms

    Classification:
        FIXED_NOISE_SUPPORT_LIMITED

    More CEM data justified:
        NO

    Scorer:
        NOT TRAINED

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
