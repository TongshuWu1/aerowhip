# Milestone 7A.1 — Existing-Data Viability Report

Generated: 2026-08-30T163122.493698Z

## 1. Why the 768-context campaign was stopped

The long teacher campaign was stopped after 42 completed contexts to test the final neural architecture before committing additional CEM compute. This run performed **zero new CEM solves** and used only durable Milestone-6A and partial Milestone-7A artifacts.

## 2. Existing data inventory

Partial 7A contains 42 completed contexts, 36 solved contexts, 605 verified successful actions, and 21504 scorer rows. Milestone 6A contains 256 contexts and 252 authoritative successes. Its final actions and outcome metrics were saved; its population candidate rows were not.

The gradient split uses 45 solved contexts, 444 successful actions, and 15893 scorer rows. The state-disjoint development split contains 246 contexts.

## 3. 6A / 7A compatibility audit

Compatibility passed: model `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`, 83-D context, final normalized 49-D action, nominal theta, variable duration [0.45, 1.80] s, 0.30-s smooth settle, 2.40-s evaluation horizon, and fixed numerical batch 2048. The maximum saved NPZ-versus-JSON 6A action difference was 2.978e-08. No old 1.20-s command semantics were used.

## 4. State-ID overlap and leakage audit

6A context overlap counts were `{'AGGREGATION_POOL': 168, 'EDGE': 0, 'EXTERNAL_DEVELOPMENT': 64, 'TEST': 0, 'TRAIN': 24, 'VALIDATION': 0}`. No 6A state overlapped frozen 7A VALIDATION, TEST, or EDGE ownership. Split leakage was absent. `TEST_PREVIOUSLY_BENCHMARKED = NO`. The final 7A TEST was neither trained on nor evaluated.

## 5. Development split

The union of eligible TRAIN-owned states was hash-ordered with seed 42 and split by state ID: 34 neural-training states and 9 internal-development states. All actions and targets for a state remain together. Existing 6A states outside TRAIN ownership are EXTERNAL DEVELOPMENT only.

## 6. Final diffusion architecture confirmation

`ConditionalActionDiffusion` is unchanged: direct 49-D normalized action, 83-D context encoder, 64-D sinusoidal time embedding, width 256, four residual MLP blocks, 100-step cosine DDPM epsilon training, EMA 0.999, and fixed-noise 25-step deterministic DDIM for 32 candidates. Parameter count: 1,249,329. No PCA/latent bottleneck or alternate architecture was introduced.

## 7. Final scorer architecture confirmation

`ManeuverOutcomeScorer` is unchanged: 132 inputs, three 256-unit SiLU layers, seven continuous physical heads and three binary heads. Parameter count: 168,202. Continuous targets use TRAIN-only normalization and Huber loss; binary targets use BCE with capped TRAIN-only positive weights. Candidate selection applies predicted scientific gates and deterministic margin/reward fallback.

## 8. Diffusion training

Training ran from scratch with AdamW 2e-4, weight decay 1e-6, gradient clip 1.0, context-balanced sampling, and EMA. It stopped at update 8000; the best EMA validation epsilon loss was 0.114847 at update 3000.

## 9. Scorer training

Training ran from scratch with AdamW 3e-4, weight decay 1e-6, outcome-stratified batches, TRAIN-only target normalization, Huber continuous losses, and BCE binary losses. It stopped at update 5000; best validation loss was 1.745611 at update 500.

## 10. Training-context oracle result

success 0.00%; feasible 0.00%; median tip error 899.798 mm; directed speed 0.919 m/s; direction 70.126 deg; UAV displacement 1958.5518 m; UAV speed 9826.3896 m/s; duration 1.8000 s; hit segments {'ACTIVE': 0, 'HOLD': 0, 'NONE': 45, 'SETTLE': 0}

Training first candidate: success 0.00%; feasible 0.00%; median tip error 962.721 mm; directed speed 0.871 m/s; direction 70.173 deg; UAV displacement 1219553.7500 m; UAV speed 6929983.5000 m/s; duration 0.4500 s; hit segments {'ACTIVE': 0, 'HOLD': 0, 'NONE': 45, 'SETTLE': 0}

Training scorer selected: success 0.00%; feasible 0.00%; median tip error 954.998 mm; directed speed 0.913 m/s; direction 71.559 deg; UAV displacement 594884.7500 m; UAV speed 3107968.5000 m/s; duration 1.8000 s; hit segments {'ACTIVE': 0, 'HOLD': 0, 'NONE': 45, 'SETTLE': 0}

The generator's mean final-coordinate clamp fraction was 99.35%. Because the training oracle was zero, the required basic sanity classification is **GENERATOR_NOT_FITTING_TEACHER_DISTRIBUTION**. This is stronger than a held-out generalization failure: the generated support does not reproduce successful actions for contexts used in gradients.

## 11. Development first-candidate result

NOT RUN

## 12. Development oracle best-of-32 result

NOT RUN

## 13. Development scorer-selected result

NOT RUN

## 14. Generator-vs-scorer decomposition

The development oracle measures whether the generator contains a physically successful candidate; the scorer-selected result measures the deployed neural choice. `P(scorer PASS | oracle contains PASS)` is NOT RUN. Classification evidence: `{'reason': 'training_oracle_below_sanity_gate'}`.

Here the decomposition stops at the generator: no generated candidate was feasible even on training contexts, so a scorer cannot rescue the candidate set. The near-total final clamp rate is a concrete sampling/distribution-boundary diagnostic and should be investigated before requesting more teacher labels.

## 15. Physical outcome metrics

The first/oracle/scorer summaries above include hard success, feasibility, median tip error, directed speed, direction error, UAV displacement/speed, maneuver duration, and ACTIVE/SETTLE/HOLD hit counts. Every number came from the frozen authoritative production simulator, not neural loss.

## 16. Scorer accuracy on diffusion candidates

NOT RUN

## 17. Teacher-count learning curve

| Fraction | Train states | Train contexts | Teacher actions | Train oracle | Dev first | Dev oracle | Dev selected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| NOT RUN — training oracle sanity gate failed | | | | | | | |

The learning curve was not run because the protocol requires an immediate stop when the training oracle is very low.

## 18. Trained neural inference latency

Over 1000 post-warmup queries, complete Python policy latency was median 27.210 ms, p90 29.030 ms, p95 29.761 ms, and maximum 51.495 ms. This includes context normalization, 32-candidate/25-step DDIM, scorer, and selection; it excludes CEM and simulation.

## 19. Whether more CEM data is justified

**NO.** Additional CEM data is justified only when the architecture learns and context coverage is the limiting factor. This decision follows the measured train/development oracle and fixed-subset learning curve, not denoising loss alone.

## 20. Explicit next recommendation

Classification: **GENERATOR_NOT_LEARNING**, with the more specific sanity finding **GENERATOR_NOT_FITTING_TEACHER_DISTRIBUTION**. Do not resume teacher generation. First review the unchanged diffusion sampling/training formulation—especially the 99% final-coordinate clamp behavior—using the existing data. No aggregation or architecture change was made here.

## 21. Final 7A TEST

**NOT EVALUATED.**

## 22. Protected fig8vertical_002

**NOT EVALUATED.**

## 23. Real hardware

**NOT EXECUTED.** All generated actions remain simulation-only.

## Final summary

    New CEM solves:
        0

    Existing solved training contexts used:
        45

    Existing successful teacher actions used:
        444

    Existing scorer rows used:
        15893

    Diffusion:
        FINAL ARCHITECTURE

    Scorer:
        FINAL ARCHITECTURE

    Train oracle best-of-32:
        0.00%

    Development first-candidate success:
        NOT RUN

    Development oracle best-of-32:
        NOT RUN

    Development scorer-selected success:
        NOT RUN

    Development feasibility:
        NOT RUN

    Learning curve:
        NOT RUN

    Median trained policy latency:
        27.210 ms

    P95 trained policy latency:
        29.761 ms

    Classification:
        GENERATOR_NOT_LEARNING

    More CEM data justified:
        NO

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
