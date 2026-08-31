# Milestone 7A.2 — Diffusion Generator Root-Cause Report

Generated: 2026-08-30T165154.107781Z

## 1. Exact observed 7A.1 failure

Milestone 7A.1 reported 99.35% final-coordinate clamping, 0% training-context oracle success, and 0% training feasibility. Direct reproduction from the saved EMA checkpoint produced 99.30% clamp, pre-clamp standard deviation 112.672, range [-363.040, 370.561], and median nearest-teacher distance 781.489.

## 2. Actual repository diffusion pipeline

The executed path is: verified `normalized_actions[N,49]` from NPZ; context-balanced sampling; normalization of the 83-D context only; uniform integer timestep sampling; cosine `alpha_bar`; forward noise `x_t=sqrt(alpha_bar_t)x_0+sqrt(1-alpha_bar_t)epsilon`; epsilon MSE; EMA checkpoint chosen by validation epsilon loss; deterministic DDIM reverse; final-only clamp; production 49-D decoder. No extra action normalization exists between disk and diffusion.

## 3. Diagnostics chosen and why

The diagnosis used: disk/loader statistics; per-timestep epsilon and implied-x0 error on known teachers; an exact-epsilon reverse control; full reverse-chain tracing; alternate pure-noise starting-timestep probes; checkpoint/EMA identity verification; and an unchanged-architecture one-context/one-action memorization experiment. Together these isolate data, equations, learned denoising, checkpointing, and inference scheduling.

## 4. Evidence from each diagnostic

- Teacher labels: 444 finite actions, mean -0.0232, std 0.2931, range [-0.7527, 0.9398], zero out of bounds.
- Exact-epsilon reverse: final maximum error 5.588e-09; reverse equations pass.
- Checkpoint: EMA update 3000 equals recorded best update 3000; tensors are finite.
- At t=99: epsilon RMSE 0.07059, but implied x0 RMSE 143.24.
- Error amplification: measured 2029.20 versus theoretical 2029.20.
- Starting the unchanged checkpoint at t=95 immediately produced std 0.2885, zero out-of-bounds coordinates, and median nearest-teacher distance 0.2785.
- The pre-repair tiny test could learn one favorable noise vector but still clamped 48.53% overall; the same checkpoint under the repaired scheduler clamps 0% and reproduces all noise-bank samples closely.

## 5. Root cause

**HIGH confidence:** the default sampler entered the isolated terminal cliff of the cosine schedule. At t=99, `alpha_bar=2.4286e-7`, so the epsilon-to-x0 conversion has gain `1/sqrt(alpha_bar)=2029.2`. The observed epsilon error is numerically small in the training loss yet becomes an x0 error of approximately 143 at the first reverse step. Every later DDIM state then remains outside the model's training distribution. The final clamp is downstream concealment, not the cause.

## 6. Why alternative hypotheses were rejected

Data corruption and action normalization were rejected by direct NPZ/loader equality and bounded label statistics. Forward/reverse inconsistency was rejected by exact-noise reconstruction. EMA/checkpoint mismatch was rejected by update and state inspection. Timestep conditioning works at t=95 and below. The production decoder is downstream of the already-failed tensor. Network capacity is not the primary defect because the exact architecture passes tiny memorization after the schedule repair.

## 7. Exact repair

The 25-step DDIM schedule now spans timesteps 95 to 0 instead of 99 to 0: `[95, 91, 87, 83, 79, 75, 71, 67, 63, 59, 55, 51, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0]`. It skips only the four-step terminal cosine cliff. Training remains 100-step epsilon prediction; inference remains deterministic 25-step DDIM; the final clamp is unchanged. No intermediate clamp was added. The existing EMA weights were retained because the demonstrated defect was exclusively in inference scheduling.

Two focused terminal-cliff regression tests pass, and the complete repository suite passes 125/125 tests.

## 8. Architecture confirmation

Architecture changed: **NO**. Diffusion family/formulation changed: **NO**. Context encoder, timestep embedding, direct 49-D state, four residual blocks, epsilon objective, action representation, context schema, production decoder, physics, and scorer are unchanged.

## 9. Tiny-data memorization result

**PASS.** With one context and one action, the unchanged final architecture trained for 10000 updates. Under the repaired schedule, clamp is 0.00%, median L2 distance to the memorized action is 0.005745, and maximum distance is 0.013727.

## 10. Teacher-versus-generated action statistics

| Statistic | Teacher | Generated pre-clamp |
|---|---:|---:|
| Mean | -0.023192 | -0.025601 |
| Std | 0.293070 | 0.292904 |
| Minimum | -0.752710 | -0.891861 |
| Maximum | 0.939766 | 0.872690 |
| Absolute p95 | 0.611049 | 0.605946 |
| Out of bounds | 0.00% | 0.00% |
| Duration mean | 0.037984 | 0.031412 |
| Duration std | 0.084074 | 0.085155 |

Generated median nearest same-context teacher distance is 0.1988; p95 is 0.3367.

## 11. Clamp fraction before and after repair

7A.1 reported: **99.35%**. Direct reproduction: **99.30%**. Post-repair over 1,440 candidates: **0.00%**. Duration has zero boundary saturation post-repair.

## 12. Training-context first-candidate result

Success: **8.89%**. Feasible: **77.78%**. Median tip error 68.112 mm; directed speed 4.101 m/s; direction error 31.037 deg; UAV displacement 0.4778 m.

## 13. Training-context oracle best-of-32 result

Success: **26.67%** (12/45). Median selected best-candidate tip error 43.701 mm; directed speed 4.158 m/s; direction error 28.449 deg; UAV displacement 0.4894 m.

## 14. Training-context feasibility

At least one feasible generated candidate exists in **100.00%** of training contexts. The selected oracle row is feasible in 66.67%. Feasible support is restored, but scientific-success coverage remains below the 50% strong sanity target.

## 15. Development oracle result

**NOT RUN.** Training oracle success was 26.67%, so `GENERATOR_SANITY_PASS` was not reached and development physics was not authorized.

## 16. Whether more CEM data is now scientifically justified

**NO.** The implementation defect is fixed, but the current checkpoint remains weak on its own training contexts. More teacher generation is not yet supported by a demonstrated context-coverage limitation.

## 17. Recommended next step

Return for methodological review of why the existing generator places only 26.67% of training contexts on the scientific-success manifold despite matching marginal action statistics. Do not resume CEM generation, scorer optimization, aggregation, or architecture replacement automatically.

## 18. Final TEST

**NOT EVALUATED.**

## 19. Protected test

`fig8vertical_002`: **NOT EVALUATED.**

## 20. Hardware

**NOT EXECUTED.**

## Final summary

    New CEM solves:
        0

    Original failure:
        99.35% generated-coordinate clamp
        0% train oracle success
        0% train feasibility

    Root cause:
        DDIM INCLUDED THE COSINE TERMINAL CLIFF AT t=99;
        EPSILON ERROR WAS AMPLIFIED 2029x INTO x0

    Root cause confidence:
        HIGH

    Architecture changed:
        NO

    Formulation changed:
        NO

    Retraining required:
        NO

    Post-repair clamp fraction:
        0.00%

    Tiny-data memorization:
        PASS

    Training first-candidate success:
        8.89%

    Training oracle best-of-32:
        26.67%

    Training oracle feasibility:
        100.00%

    Development oracle best-of-32:
        NOT RUN

    Result:
        IMPLEMENTATION_BUG_FIXED_BUT_MODEL_WEAK

    More CEM data justified:
        NO

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
