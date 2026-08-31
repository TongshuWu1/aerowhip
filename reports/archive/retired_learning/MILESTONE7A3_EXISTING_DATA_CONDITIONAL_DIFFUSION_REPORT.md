# Milestone 7A.3 — Existing-Data Scaling and Conditional Diffusion Report

Generated: 2026-08-30T174531.609210Z

## 1. Why no new CEM was run

The experiment used only durable partial-7A and Milestone-6A authoritative actions. The 6A benchmark already contained paid-for TRAIN, AGGREGATION_POOL, and EXTERNAL_DEVELOPMENT coverage. Running another optimizer before consuming these labels would not distinguish data coverage from conditional-learning failure. New CEM solves: **0**.

## 2. 7A.2 scheduler repair confirmation

The final generator uses the repaired deterministic 25-step DDIM schedule `[95, 91, 87, 83, 79, 75, 71, 67, 63, 59, 55, 51, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0]`. It begins at t=95 and ends at t=0, avoiding the t=99 cosine terminal cliff. Training remains a 100-step cosine DDPM epsilon objective. No scheduler, formulation, action representation, or architecture change was made in 7A.3.

## 3. Existing-data inventory

The deduplicated corpus contains 202 gradient-training contexts (680 verified successful actions), 22 internal denoising-validation contexts (113 actions), and 64 external-development contexts. Context identity includes state, target, direction, theta, bank, and schema; duplicate action labels were removed at 1e-7 normalized-coordinate precision while all provenance was retained.

## 4. Ownership and leakage audit

Only 7A TRAIN and AGGREGATION_POOL state IDs entered gradients. AGGREGATION_POOL was explicitly consumed as training data for this experiment. Internal validation was split 90/10 by state ID with seed 42. EXTERNAL_DEVELOPMENT contributed no gradients, normalizer fitting, early stopping, or checkpoint selection. 7A VALIDATION, TEST, EDGE-reserved, and `fig8vertical_002` contributed zero gradient rows. State-level leakage: **NONE**.

## 5. Distinct states, contexts, and actions

- Gradient-training states: 185
- Gradient-training contexts: 202
- Gradient-training successful actions: 680
- Internal validation states: 21
- Internal validation contexts: 22
- External-development contexts: 64

The context-balanced sampler first samples a context uniformly and then one successful action uniformly within that context. Multi-solution partial-7A contexts therefore do not dominate single-solution 6A contexts.

## 6. External-development definition

The primary physics-development set is the immutable set of 64 compatible Milestone-6A EXTERNAL_DEVELOPMENT contexts. These are canonical-state target-variation cases, each with an authoritative CEM success. They remained fully gradient-free and were not used for denoising validation or checkpoint selection.

## 7. Final diffusion architecture confirmation

Architecture is unchanged: 83-D context encoder `83→256 SiLU→256`; sinusoidal 64-D timestep embedding projected to 256; 49-D noisy-action projection; four `LayerNorm→256→512→SiLU→256` residual MLP blocks; and a 49-D epsilon output. The model has the existing approximately 1.249M parameters, direct normalized [49] diffusion, EMA 0.999, and no PCA/latent/transformer bottleneck.

## 8. Full-model training

The FULL model was initialized from scratch and trained with AdamW (`lr=2e-4`, weight decay `1e-6`), batch up to 1024, gradient clip 1.0, and EMA 0.999. It ran 7500 updates; best internal-validation epsilon loss was 0.085504 at update 2500. External development was not consulted during training.

## 9. Generated action-distribution sanity

Pre-clamp generated actions have mean -0.02838, standard deviation 0.28828, absolute p95 0.59619, and range [-1.56186, 0.81402]. The final clamp fraction is 0.05%; normalized duration ranges from -0.3324 to 0.2310. The terminal-cliff pathology did not recur.

## 10. Training first-candidate result

First-candidate scientific success on 202 gradient-training contexts is 2.48%; feasibility is 60.40%. Median first-candidate tip error is 104.29 mm, directed speed 4.029 m/s, direction error 28.54°, UAV displacement 0.489 m, UAV speed 2.225 m/s, and duration 1.124 s.

## 11. Training oracle best-of-32 result

Training oracle best-of-32 scientific success is 18.32%, compared with 26.67% in Milestone 7A.2.

## 12. Training feasible-support result

At least one feasible candidate exists for 100.00% of training contexts. Candidate-level feasibility is 66.58%. Median oracle tip error is 50.08 mm, directed speed 4.293 m/s, direction error 28.36°, UAV displacement 0.492 m, and UAV speed 2.197 m/s. Oracle hit segments are {'ACTIVE': 86, 'SETTLE': 15, 'HOLD': 0, 'NONE': 101}.

## 13. External-development first-candidate result

External-development first-candidate scientific success is 4.69%; feasibility is 9.38%. Median first-candidate tip error is 87.75 mm, directed speed 4.020 m/s, direction error 23.97°, UAV displacement 0.559 m, UAV speed 2.393 m/s, and duration 1.137 s.

## 14. External-development oracle best-of-32

External-development oracle best-of-32 success is 60.94%. At least one feasible candidate exists for 100.00%; candidate-level feasibility is 45.51%. Median oracle tip error is 38.70 mm, directed speed 4.278 m/s, direction error 25.25°, UAV displacement 0.485 m, UAV speed 2.244 m/s, and maneuver duration 1.124 s. Oracle hit segments are {'ACTIVE': 43, 'SETTLE': 12, 'HOLD': 0, 'NONE': 9}.

## 15. Amortization conditional on CEM success

All 64 external-development references are authoritative CEM successes. Therefore `P(diffusion oracle PASS | CEM PASS)` equals the diffusion external-development oracle: **60.94%**.

## 16. Correct-versus-shuffled context experiment

Using identical fixed noise, correct-context oracle success is 60.94%; deterministically shuffled conditioning evaluated against the original task yields 25.00%. The median matched-noise action change is 0.2529. Median oracle task cost changes from -593.4154 to 6.5885.

## 17. Target-conditioning experiment

Across 16 same-state/different-target pairs, the median matched-noise generated-action distance is 0.3646, versus median teacher-action distance 0.4638. Teacher/generated change-magnitude correlation is 0.9647313369782033. Own-target oracle success is 3.12%; cross-swapping each pair's generated candidates to the other target gives 0.00%. Classification: **ACTIVE**.

## 18. Initial-state-conditioning experiment

Across 16 same-target/different-physical-state pairs, the median matched-noise generated-action distance is 0.1656, versus median teacher-action distance 0.4364. Teacher/generated change-magnitude correlation is -0.04867883714438381. Own-state oracle success is 40.62%; cross-swapping each pair's generated candidates to the other physical state gives 12.50%. Classification: **ACTIVE**.

## 19. Context-sensitivity summary

Correct conditioning exceeds shuffled conditioning by 35.94 percentage points. Median generated action changes are 0.3646 for target changes, 0.1656 for state changes, and 0.2529 for shuffled context. This separates mere marginal-distribution matching from behaviorally useful conditioning.

## 20. Existing-data learning curve

| Model | States | Contexts | Labels | Train oracle | Dev first | Dev oracle | Dev any feasible | Clamp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE | 41 | 45 | 154 | 0.00% | 0.00% | 1.56% | 54.69% | 0.77% |
| MID | 92 | 101 | 331 | 6.67% | 3.12% | 43.75% | 95.31% | 0.82% |
| FULL | 185 | 202 | 680 | 13.33% | 4.69% | 60.94% | 100.00% | 0.05% |

## 21. Does performance scale with context count?

The classification logic treats a ≥5-point FULL-over-BASE development-oracle improvement, without a material FULL regression below MID, as a clear scaling signal. The measured result is recorded in `classification_evidence.json` and supports the final evidence-based classification below.

## 22. Neural inference latency

For context normalization plus deterministic 32-candidate, 25-step DDIM generation, median latency is 26.31 ms, p90 28.22 ms, p95 29.33 ms, and maximum 63.27 ms across 1000 warmed queries. Scorer, simulator, and CEM are excluded.

## 23. Whether more CEM data is justified

**NO**. BASE→MID→FULL development performance does scale, which is encouraging, but the FULL generator still reaches only 18.32% oracle success on its own broad training contexts—below the earlier 26.67% sanity result. The milestone explicitly says weak own-training fit overrides a scaling-only argument for purchasing more labels. No CEM was launched regardless of this result.

## 24. Recommended next step

Classification: **GENERATOR_STILL_WEAK**. Conditioning is demonstrably active and external target-only performance is useful, so the next review should isolate why the same model underfits the heterogeneous state/target training corpus—for example, per-source/group oracle breakdown and conditional denoising error—before buying more labels or resuming scorer work. No next experiment was started automatically.

## 25. Scorer

**NOT TRAINED / NOT EVALUATED.**

## 26. Final TEST

**NOT EVALUATED.**

## 27. Protected test

`fig8vertical_002`: **NOT EVALUATED.**

## 28. Hardware

Real hardware: **NOT EXECUTED.** All maneuvers remain simulation-only.

## 29. Regression verification

The focused 7A.3/diffusion contract tests pass 17/17. The complete repository regression suite passes **128/128** tests. The audit locks the repaired t95→0 schedule and statically confirms that this runner contains neither a CEM-optimizer call nor a scorer-training call.

## Final summary

    New CEM solves:
        0

    Scheduler:
        REPAIRED t95->0 DDIM

    Diffusion architecture:
        UNCHANGED FINAL ARCHITECTURE

    Distinct training states:
        185

    Training contexts:
        202

    Successful teacher actions:
        680

    External-development contexts:
        64

    Clamp fraction:
        0.05%

    Training first-candidate success:
        2.48%

    Training oracle best-of-32:
        18.32%

    Training contexts with feasible candidate:
        100.00%

    External-development first-candidate success:
        4.69%

    External-development oracle best-of-32:
        60.94%

    Policy oracle success given CEM success:
        60.94%

    Correct-context oracle:
        60.94%

    Shuffled-context oracle:
        25.00%

    Target conditioning:
        ACTIVE

    State conditioning:
        ACTIVE

    Learning curve:

        BASE contexts = 45
        dev oracle = 1.56%

        MID contexts = 101
        dev oracle = 43.75%

        FULL contexts = 202
        dev oracle = 60.94%

    Median generator latency:
        26.31 ms

    P95 generator latency:
        29.33 ms

    Classification:
        GENERATOR_STILL_WEAK

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
