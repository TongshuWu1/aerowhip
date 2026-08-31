# Milestone 7A.5 — Frozen Diffusion Candidate-Support and Latency Study

Artifact root: `data\policy_training\diffusion_candidate_support_v1\2026-08-30T184254.122785Z`  
Completed UTC: `2026-08-30T19:01:26.607719+00:00`

## 1. Why generator training was frozen

Milestone 7A.4 showed a literal-prefix improvement from 39.06% at 32 candidates to 64.06% at 128 candidates. This milestone isolates sampling support from representation learning: the 7A.3 FULL flat-conditioning EMA generator, its context/action contracts, DDIM scheduler, and production simulator were frozen. No CEM, diffusion training, scorer work, dataset generation, or architecture change occurred.

## 2. Selected 7A.3 generator confirmation

The evaluated generator is `BASELINE_FLAT_CONDITIONING`, checkpoint SHA-256 `49727d097ca95997645014b510c4a4b6d1738fa978b9067e65bdbb5da4dbb01b`, with 1,249,329 trainable parameters. It maps normalized 83-D contexts to direct normalized 49-D actions through the repaired deterministic 25-step DDIM timetable `95 -> 0` with `eta=0`. The experimental structured-FiLM checkpoint was not loaded.

## 3. Reproduction of corrected 7A.4 32/64/128 result

The hard reproduction gate **passed exactly**. Expected and observed oracle rates were 39.0625%, 56.2500%, and 64.0625% at N=32, 64, and 128; every difference was 0.0. The generated-action clamp fraction in this reproduction was 0.11%.

## 4. Exact primary development context set

`primary_development_manifest.json` freezes all 64 context IDs, state IDs, targets, directions, and ownership labels. Composition: EXTERNAL_DEVELOPMENT / canonical: 32, TRAIN / training: 32. There are 33 unique physical state IDs. No substitution was made after the manifest was written.

## 5. Heterogeneous state diagnostic set

`heterogeneous_state_manifest.json` freezes 64 unique TRAIN-owned state/context pairs, with one target per physical state. Existing state distance was stratified into LOW=21, MEDIUM=22, and HIGH=21; no TEST or protected context appears.

## 6. Exact nested 1024 noise-bank construction

The primary bank is one persisted `[1024,49]` Gaussian array with SHA-256 `d53acfd097929178f265f9547aa1ea73b2782a9d832db0ccfa8301398d3891af`. Exact recovery of the 7A.4 first 128 vectors is `TRUE`. Rows 128:1024 were deterministically precommitted with seed 74128. All N values are literal prefixes of this one bank. Three independent `[512,49]` banks were also persisted before physics with seeds 750501, 750502, 750503; none was selected by performance.

## 7. Production physics evaluation contract

For each context, the maximum bank was generated once, decoded as the production normalized `[49]` action, and evaluated once. Execution used 16 acceleration knots plus `T_maneuver in [0.45,1.80] s`, ACTIVE -> 0.30-s analytic SETTLE -> HOLD, `T_evaluation=2.40 s`, the frozen `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`, and cyclic padding to the fixed numerical batch of 2048. Prefix results were sliced from saved candidate/outcome tensors; no prefix was regenerated or resimulated. Scientific gates were unchanged.

## 8. Primary candidate-count oracle curve

| N | Oracle success | Contexts with any feasible candidate | Candidate-level feasibility |
|---:|---:|---:|---:|
| 32 | 39.06% | 100.00% | 53.86% |
| 64 | 56.25% | 100.00% | 50.61% |
| 128 | 64.06% | 100.00% | 49.63% |
| 256 | 76.56% | 100.00% | 50.18% |
| 512 | 79.69% | 100.00% | 49.17% |
| 1024 | 82.81% | 100.00% | 49.67% |

The primary curve rises 43.75 percentage points from N=32 to N=1024.

## 9. Heterogeneous-state candidate-count oracle curve

| N | Oracle success | Contexts with any feasible candidate | Candidate-level feasibility |
|---:|---:|---:|---:|
| 32 | 18.75% | 100.00% | 64.36% |
| 64 | 26.56% | 100.00% | 63.23% |
| 128 | 35.94% | 100.00% | 62.50% |
| 256 | 42.19% | 100.00% | 62.15% |
| 512 | 53.12% | 100.00% | 61.45% |
| 1024 | 60.94% | 100.00% | 61.94% |

The heterogeneous curve rises 42.19 points but remains 21.88 points below primary at N=1024. Every context in both sets contains at least one feasible candidate at every reported N; the missing support is joint task success, not absence of safe motion.

## 10. First-success-index distribution

Among the 53/64 primary contexts solved by N=1024, the one-based first-success index has median 39, p75 98, p90 226.0, p95 280.8, and maximum 1010. For the 39/64 heterogeneous contexts solved, the corresponding values are median 81, p75 304, p90 531.4, p95 579.0, and maximum 781.

## 11. Successful-candidate density

Primary contexts contain a mean 15.23 and median 14.0 successes per 1024 candidates: mean density 1.49%, median density 1.37%, p10 0.00%, p90 3.12%. Heterogeneous contexts contain mean 5.70, median 1.5, with mean density 0.56%. Success is therefore sparse even when a context is solvable.

## 12. Feasible-candidate density

Primary candidate feasibility has mean 49.67%, median 42.14%, p10 32.72%, and p90 78.13%. Heterogeneous feasibility is higher: mean 61.94%, median 66.89%. Sampling more candidates exposes rare joint success rather than merely finding the first feasible action.

## 13. Per-gate support progression

Primary context-level support is:

| N | Any feasible | Distance | Directed speed | Direction | Tip first | Joint success |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 100.00% | 73.44% | 100.00% | 100.00% | 87.50% | 39.06% |
| 64 | 100.00% | 79.69% | 100.00% | 100.00% | 92.19% | 56.25% |
| 128 | 100.00% | 84.38% | 100.00% | 100.00% | 96.88% | 64.06% |
| 256 | 100.00% | 85.94% | 100.00% | 100.00% | 98.44% | 76.56% |
| 512 | 100.00% | 92.19% | 100.00% | 100.00% | 98.44% | 79.69% |
| 1024 | 100.00% | 95.31% | 100.00% | 100.00% | 98.44% | 82.81% |

At N=1024, individual speed and direction support reach 100%, tip-first reaches 98.44%, and distance reaches 95.31%, yet joint success is 82.81%. The remaining issue is overlap of gates in the same candidate.

## 14. N=1024 failure-context analysis

There are 11 primary and 25 heterogeneous contexts with zero joint success. Counts with no candidate passing an individual gate are primary `{'distance': 3, 'directed_speed': 0, 'direction': 0, 'tip_first': 1, 'feasible': 0}` and heterogeneous `{'distance': 8, 'directed_speed': 0, 'direction': 0, 'tip_first': 0, 'feasible': 0}`. Detailed best-candidate physical values and exact IDs are preserved in `failure_contexts_1024.json`. Many failures have candidates that individually pass distance, speed, direction, tip-first, and feasibility, but no single candidate passes all gates; UAV displacement is a frequent incompatibility in the heterogeneous rows.

| Set | Context | State | Best tip mm | Best speed m/s | Best direction deg | Best UAV displacement m | No candidate passes |
|---|---|---|---:|---:|---:|---:|---|
| Primary | `m7a3_5d407907f9ce6dc5cea4` | `training:2332` | 29.0 | 4.392 | 24.73 | 0.5435 | joint overlap only |
| Primary | `m7a3_ae02570e1959529749d0` | `training:0220` | 72.3 | 4.054 | 32.64 | 0.5013 | distance |
| Primary | `m7a3_556ac1d3535f0812069a` | `training:0745` | 67.4 | 4.137 | 28.62 | 0.5363 | joint overlap only |
| Primary | `m7a3_4839c23b035a318f944b` | `training:3178` | 28.5 | 4.267 | 22.72 | 0.5687 | joint overlap only |
| Primary | `m7a3_cbe16394b6a1f73c2174` | `training:0886` | 64.3 | 4.276 | 21.40 | 0.4943 | distance, tip_first |
| Primary | `m7a3_939f0a415dad39651f55` | `training:1682` | 79.0 | 4.109 | 26.65 | 0.5226 | distance |
| Primary | `m7a3_b0d16bf202f85011b8e8` | `training:2848` | 57.5 | 4.141 | 26.32 | 0.6816 | joint overlap only |
| Primary | `m7a3_2bca21697c868f5be228` | `training:3976` | 38.7 | 4.218 | 21.35 | 0.5325 | joint overlap only |
| Primary | `m7a3_0a42614b4d65fc229a8f` | `training:0443` | 1107.9 | 1.064 | 69.62 | 38403528.0000 | joint overlap only |
| Primary | `m7a3_068d801ab2d77804cc90` | `training:3006` | 62.0 | 4.365 | 18.77 | 0.5817 | joint overlap only |
| Primary | `m7a3_6ec170df67be758cdc84` | `training:2393` | 61.0 | 4.154 | 30.98 | 0.4799 | joint overlap only |
| Heterogeneous | `m7a3_c42d7de7a83aac0fb078` | `training:3391` | 106.7 | 3.852 | 26.06 | 0.5187 | distance |
| Heterogeneous | `m7a3_4839c23b035a318f944b` | `training:3178` | 28.5 | 4.267 | 22.72 | 0.5687 | joint overlap only |
| Heterogeneous | `m7a3_867e118695885584e333` | `training:0458` | 54.2 | 4.304 | 27.47 | 0.6117 | joint overlap only |
| Heterogeneous | `m7a3_2bca21697c868f5be228` | `training:3976` | 38.7 | 4.218 | 21.35 | 0.5325 | joint overlap only |
| Heterogeneous | `m7a3_854adfd01521135e7d37` | `training:1243` | 51.5 | 4.174 | 28.34 | 0.4806 | joint overlap only |
| Heterogeneous | `m7a3_1f9be8b786f79530c8b3` | `training:1222` | 88.7 | 4.208 | 21.22 | 0.6507 | distance |
| Heterogeneous | `m7a3_ffc5cc69f4badcb35efb` | `training:1094` | 115.5 | 4.167 | 27.82 | 0.4501 | distance |
| Heterogeneous | `m7a3_b0d16bf202f85011b8e8` | `training:2848` | 57.5 | 4.141 | 26.32 | 0.6816 | joint overlap only |
| Heterogeneous | `m7a3_5ecf6c1805713a22b3f6` | `training:2782` | 12.3 | 4.567 | 25.65 | 0.5786 | joint overlap only |
| Heterogeneous | `m7a3_5d407907f9ce6dc5cea4` | `training:2332` | 29.0 | 4.392 | 24.73 | 0.5435 | joint overlap only |
| Heterogeneous | `m7a3_1b8c0835a3243348603b` | `training:1198` | 81.7 | 4.094 | 28.74 | 0.5139 | distance |
| Heterogeneous | `m7a3_af91a005bcf2faccbd74` | `training:1401` | 77.4 | 4.240 | 33.74 | 0.5542 | joint overlap only |
| Heterogeneous | `m7a3_69f19925a1d3800dc636` | `training:0606` | 56.9 | 4.122 | 25.44 | 0.4715 | distance |
| Heterogeneous | `m7a3_2bdda35e28831c9af559` | `training:1583` | 37.9 | 4.702 | 22.61 | 0.8383 | joint overlap only |
| Heterogeneous | `m7a3_10ae946ea69c6e1a045a` | `training:1877` | 21.3 | 4.263 | 21.96 | 0.5960 | joint overlap only |
| Heterogeneous | `m7a3_8deb211c2e316983b2d2` | `training:2516` | 68.2 | 4.214 | 21.29 | 0.5760 | distance |
| Heterogeneous | `m7a3_754599db04faa2d33ec3` | `training:2367` | 55.9 | 4.406 | 31.54 | 0.5764 | joint overlap only |
| Heterogeneous | `m7a3_b735a0588f54c01a206f` | `training:2509` | 50.5 | 4.517 | 28.92 | 0.4609 | joint overlap only |
| Heterogeneous | `m7a3_b353a1a4b7e769ef9cbe` | `training:0956` | 35.9 | 4.539 | 33.19 | 0.5247 | joint overlap only |
| Heterogeneous | `m7a3_4de4dcf99e2c818c39bb` | `training:3344` | 44.9 | 4.310 | 28.42 | 0.5229 | joint overlap only |
| Heterogeneous | `m7a3_b29c91651e86c6146f2a` | `training:2790` | 10.4 | 4.651 | 36.29 | 0.5605 | joint overlap only |
| Heterogeneous | `m7a3_e9b7219388e23557a8bb` | `training:0298` | 72.1 | 4.230 | 19.73 | 0.5895 | joint overlap only |
| Heterogeneous | `m7a3_8d8ed15d06b12adbaec4` | `training:2401` | 67.6 | 4.249 | 29.39 | 0.5023 | distance |
| Heterogeneous | `m7a3_18e155343d1ee4373446` | `training:1531` | 14.8 | 4.577 | 30.37 | 0.5648 | joint overlap only |
| Heterogeneous | `m7a3_fe94caf0aa7a7b34b598` | `training:0780` | 64.7 | 4.294 | 25.96 | 0.5090 | distance |

## 15. Independent-noise-bank robustness

| N | Four-bank mean | Std | Minimum | Maximum | Union (analysis only) |
|---:|---:|---:|---:|---:|---:|
| 32 | 28.12% | 12.60 pp | 14.06% | 42.19% | 67.19% |
| 64 | 50.00% | 7.97 pp | 40.62% | 59.38% | 76.56% |
| 128 | 62.50% | 4.28 pp | 57.81% | 68.75% | 82.81% |
| 256 | 73.44% | 2.21 pp | 70.31% | 76.56% | 85.94% |
| 512 | 81.25% | 1.91 pp | 79.69% | 84.38% | 87.50% |

## 16. Whether fixed bank identity materially changes conclusions

Bank identity matters at small N: N=32 spans 14.06–42.19%. Variability contracts with sampling; N=512 spans 79.69–84.38%, standard deviation 1.91 percentage points. Thus no bank was cherry-picked and the large-N conclusion is not a one-bank anomaly. The union is analysis-only and is not presented as a deployment policy.

## 17. Generation latency by N

| N | Mean wall ms | Median wall ms | P90 wall ms | P95 wall ms | Max wall ms | Median GPU ms | Peak allocated MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 24.85 | 24.78 | 26.48 | 27.22 | 32.03 | 24.76 | 14.08 |
| 64 | 51.23 | 51.20 | 53.98 | 54.62 | 56.89 | 51.17 | 14.10 |
| 128 | 76.77 | 76.38 | 79.68 | 80.65 | 95.60 | 76.36 | 14.30 |
| 256 | 101.68 | 101.23 | 105.27 | 105.96 | 113.77 | 101.21 | 14.71 |
| 512 | 125.45 | 125.18 | 128.71 | 129.80 | 134.81 | 125.15 | 15.52 |
| 1024 | 149.14 | 148.85 | 152.98 | 154.84 | 183.85 | 148.82 | 17.14 |

These timings include context normalization, deterministic 25-step DDIM, final bounded-action preparation, and the deterministic hierarchical chunk path, but exclude simulator and scorer.

## 18. CUDA memory by N

| N | Peak allocated MiB | Peak reserved MiB | Candidate chunks |
|---:|---:|---:|---|
| 32 | 14.08 | 30.00 | 32 |
| 64 | 14.10 | 30.00 | 32 + 32 |
| 128 | 14.30 | 30.00 | 32 + 32 + 64 |
| 256 | 14.71 | 30.00 | 32 + 32 + 64 + 128 |
| 512 | 15.52 | 30.00 | 32 + 32 + 64 + 128 + 256 |
| 1024 | 17.14 | 30.00 | 32 + 32 + 64 + 128 + 256 + 512 |

The hierarchical path preserves exact noise-vector order and bounds peak allocation below 18 MiB. A focused whole-batch-versus-chunked check on 128 candidates passed: bounded max absolute difference `8.941e-07` against tolerance `2.0e-05`. No extra memory-driven subdivision was needed.

## 19. Recommended practical N

No evaluated N reaches the requested 90% primary-support criterion. For continued *offline analysis* the recommended sampled prefix is N=1024, because the N=512 -> 1024 increment is 3.12% on primary and 7.81% on heterogeneous states. This does **not** change the production candidate count. The 1000-query precision run at N=1024 measured median 147.20 ms and p95 153.90 ms, above the prior 100-ms p95 engineering target.

At N=1024, median successful-oracle physical metrics are tip distance 32.45 mm, directed speed 4.347 m/s, direction error 24.92 deg, UAV displacement 0.4900 m, UAV speed 2.118 m/s, command acceleration 14.425 m/s^2, maneuver duration 1.116 s, and hit time 1.100 s. The 5th-percentile gate margins are saved in `recommended_candidate_count.json`.

Successful-oracle physical metrics across all prefixes are:

| Set | N | Tip mm | Directed speed m/s | Direction deg | UAV displacement m | UAV speed m/s | Command accel m/s^2 | Maneuver s | Hit s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Primary | 32 | 35.97 | 4.221 | 25.93 | 0.4763 | 2.104 | 14.103 | 1.111 | 1.090 |
| Primary | 64 | 38.21 | 4.269 | 26.12 | 0.4807 | 2.128 | 14.296 | 1.118 | 1.100 |
| Primary | 128 | 41.27 | 4.321 | 26.70 | 0.4835 | 2.124 | 14.371 | 1.116 | 1.090 |
| Primary | 256 | 35.72 | 4.331 | 26.39 | 0.4850 | 2.137 | 14.535 | 1.118 | 1.100 |
| Primary | 512 | 34.95 | 4.353 | 25.34 | 0.4885 | 2.105 | 14.236 | 1.118 | 1.100 |
| Primary | 1024 | 32.45 | 4.347 | 24.92 | 0.4900 | 2.118 | 14.425 | 1.116 | 1.100 |
| Heterogeneous | 32 | 41.92 | 4.194 | 29.09 | 0.4786 | 2.137 | 14.553 | 1.102 | 1.085 |
| Heterogeneous | 64 | 42.54 | 4.201 | 28.11 | 0.4687 | 2.156 | 14.469 | 1.106 | 1.090 |
| Heterogeneous | 128 | 39.13 | 4.279 | 28.11 | 0.4799 | 2.156 | 14.622 | 1.100 | 1.090 |
| Heterogeneous | 256 | 37.30 | 4.331 | 27.62 | 0.4860 | 2.192 | 14.532 | 1.104 | 1.090 |
| Heterogeneous | 512 | 36.27 | 4.401 | 27.45 | 0.4877 | 2.111 | 14.755 | 1.110 | 1.090 |
| Heterogeneous | 1024 | 35.24 | 4.357 | 27.33 | 0.4894 | 2.091 | 14.514 | 1.121 | 1.090 |

At N=1024, successful primary hits are `{'ACTIVE': 49, 'HOLD': 0, 'NONE': 0, 'SETTLE': 4}`. Margins are positive distance from each hard limit:

| Gate | Median margin | P05 margin | Unit |
|---|---:|---:|---|
| tip distance | 17.554 | 1.766 | mm |
| directed speed | 0.347 | 0.096 | m/s |
| direction | 5.082 | 0.915 | deg |
| UAV displacement | 10.025 | 0.715 | mm |
| UAV speed | 0.882 | 0.570 | m/s |
| command acceleration | 5.575 | 4.228 | m/s^2 |

## 20. Candidate-support classification

**STATE_SUPPORT_LIMITED**. The primary distribution contains substantial sparse support, and support continues increasing through 1024. However, 82.81% remains below the 90% sufficiency gate, the heterogeneous-state result is only 60.94%, and large-N latency is not within the immediate-query target. The dominant distinction is state support; small-N bank sensitivity is secondary and largely contracts by N=512.

## 21. Whether scorer training is justified

**NO.** The task authorizes scorer training only for `CANDIDATE_SUPPORT_SUFFICIENT`; that condition was not met. No scorer was trained or used.

## 22. Whether more CEM data is justified

**NO from this experiment alone.** Candidate-count failure is not evidence that more labels will repair conditional state support. This milestone ran zero CEM solves and makes no automatic teacher-expansion decision.

## 23. Recommended next step

Return for methodological review of state-conditioned support. The useful next question is how to improve support over heterogeneous initial states without conflating the issue with bank size or scorer ranking. Do not start scorer training, CEM generation, retraining, or architecture changes automatically.

## 24. Final TEST

**NOT EVALUATED.**

## 25. Protected test

`fig8vertical_002`: **NOT EVALUATED.**

## 26. Hardware

**NOT EXECUTED.** All results are simulation-only.

## Final summary

    New CEM solves:
        0

    Diffusion training:
        NONE

    Generator:
        FROZEN 7A.3 FLAT MODEL

    Primary context count:
        64

    State-diagnostic context count:
        64

    Primary oracle:

        N=32:
            39.06%

        N=64:
            56.25%

        N=128:
            64.06%

        N=256:
            76.56%

        N=512:
            79.69%

        N=1024:
            82.81%

    Heterogeneous-state oracle:

        N=32:
            18.75%

        N=64:
            26.56%

        N=128:
            35.94%

        N=256:
            42.19%

        N=512:
            53.12%

        N=1024:
            60.94%

    First-success-index:

        median = 39
        p90 = 226.0
        p95 = 280.8

    Noise-bank variability:
        N=512 mean 81.25%, std 1.91 pp, range 79.69–84.38%

    Recommended candidate count:
        1024 FOR ANALYSIS; PRODUCTION COUNT NOT CHANGED

    Median latency at recommended N:
        147.20 ms

    P95 latency:
        153.90 ms

    Candidate-support result:
        STATE_SUPPORT_LIMITED

    Scorer training justified:
        NO

    More CEM data justified:
        NO

    Architecture changed:
        NO

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
