# M0 physical-flight evaluation — 13 September 2026

**Five clean executions of the same slower-braking CSV. No model fitting is used in these scores.**

Target: (1.25, 0, 1.25) m. Commanded launch: (0, 0, 1.4) m. Raw global coordinates are retained.

The fixed strike time is **1.117248246 s** after CSV onset, taken from the original saved M0 forecast before scoring these takes. It is the planner’s predicted target-sphere entry time, not a retrospectively selected measured time. The physical target metric has no binary radius threshold.

| Take | Error at planned strike (cm) | Closest distance (cm) | Closest time (s) | Drone forecast RMSE (cm) | Tip forecast RMSE (cm) |
|---|---:|---:|---:|---:|---:|
| M0_001 | 15.24 | 12.77 | 1.1295 | 16.82 | 22.73 |
| M0_002 | 26.91 | 23.54 | 1.1363 | 22.22 | 27.26 |
| M0_003 | 20.44 | 17.27 | 1.1320 | 18.68 | 24.33 |
| M0_004 | 33.75 | 7.48 | 1.1699 | 14.69 | 21.28 |
| M0_005 | 24.27 | 12.59 | 1.1471 | 14.83 | 21.44 |

## Aggregate results

| Metric | Equal-take mean ± sample SD (cm), n=5 |
|---|---:|
| Error at planned strike time | 24.12 ± 6.94 |
| Closest approach, 0–1.5 s | 14.73 ± 6.02 |
| Quadrotor forecast RMSE, 0–1.5 s | 17.45 ± 3.13 |
| Tip forecast RMSE, 0–1.5 s | 23.41 ± 2.48 |
| All 10 cable markers forecast RMSE, 0–1.5 s | 21.18 ± 3.21 |

RMSE is the square root of the mean squared **3D Euclidean** error, computed within each take and then averaged equally across takes. The SD describes variation across five takes; it is not a confidence interval. Marker RMSE pools the 10 observed marker sites within each take, excludes the constructed attachment site, and retains native gaps.

## Interpretation

Every take is short of the target in x at the planned strike time. Closest approach follows later, between 1.129 and 1.170 s. The smaller closest-distance error therefore does not imply accurate strike timing. Residual lateral and vertical error remains at closest approach. These observations describe geometry; they do not establish impact, impact energy, or a causal explanation of the model error.

The original forecast assumes the planned launch state. Its errors include differences in actual initial state as well as model error. The later M0/M1 comparison must use the same causal measured initialization, schedules and masks for both models and must be reported separately.

## Data and timing audit

- All 122 dynamic CSV packets are present in each controller log; packet reception timing spread is 0.265–0.432 ms.
- OptiTrack rigid body: cf_3. Cable: cable1:c1 through c10 in numeric order. Unlabeled markers are ignored.
- Tip tracking is 100% complete over 0–1.5 s in all five takes. All-marker coverage is at least 99.93%. No overlength segments were found in this interval under the existing preparation rule.
- Closest approach uses adjacent valid 100 Hz observations with piecewise-linear interpolation. No interpolation bridges missing samples or large timestamp gaps.
- Clock offsets are estimated by matching the two measured quadrotor position streams without spatial fitting. Alignment residuals are 1.24–1.58 cm; estimates include logging latency. Half-record offset spread reaches 9.2 ms for take 003.
- Shifting the estimated alignment by ±10 ms gives an equal-take mean fixed-time error range of 20.45–28.65 cm. This is a sensitivity check, not a statistical uncertainty interval. Closest approach is substantially less sensitive to a constant time shift because the encounter remains inside the fixed window.
- Recordings 001, 002 and 004 end before the final 7.133 s hold finishes. This does not truncate the reported 0–1.5 s endpoints; no full-duration recovery score is claimed.

## M1 preparation

The original split is adaptation 001/002/004 and operational validation 003/005. Take 002 has four missing tip observations and invalid adjacent-marker geometry at approximately −0.75 to −0.72 s. It fails the unchanged one-second causal initialization requirement. Its M0 task measurements above remain valid. The operator requested retaining this take. The revised, explicit initialization policy omits masked observations independently for each cable node from the weighted quadratic regression, retains the one-second history and 0.02 s weighting constant, requires at least 80% coverage and 11 valid samples per node, and requires the final initial-state position to be observed. Take 002 retains 96.04% or more node coverage; all other takes retain 100%. No missing positions are synthesized. The same initializer is used for both M0 and M1 in the postflight model comparison. The original split is preserved.

The existing staged update uses current whip training data plus the retained preliminary training takes, with training-only stopping. M0 is preserved. M1 enables a cable residual absent in M0, so the comparison includes a model-capacity change. This update batch is not the final independent physical test.

## Artifacts

- [Per-take metrics](per_take.csv), [complete numerical report](paper_metrics.json), and [source checksums](source_hashes.json).
- [Trajectory and forecast plots (PDF)](M0_frozen_forecast.pdf) / [editable SVG](M0_frozen_forecast.svg).
- [Target-error chart (PDF)](M0_target_errors.pdf) / [editable SVG](M0_target_errors.svg).

Generated on Windows using the project runtime. These are measured update-batch results, with the limitations above; no M1 result is implied.

## Verification and fitting status

47 targeted tests passed; six legacy integration tests were skipped because the old local adp0 fixture is unavailable. One initial pytest invocation reported a cache-directory permission warning. The CUDA fit ran on Windows / RTX 4080 and stopped before cable residual training because the full-rollout gradient check failed at the selected physical iterate. Completed stages and the failure are preserved. The bounded audit retained physical update 6, the lowest-training-loss saved iterate passing the unchanged gradient check; updates 3, 4 and 7 failed. The completed quadrotor fit is reused exactly in `runs/adaptation/M1-paper-20260913-verified-physical`, where cable residual training is continuing. No validation outcome entered this choice. Its frozen inputs are in `runs/adaptation/M1-paper-20260913`; the failed strict-history preparation remains preserved separately. No fit result or improvement is claimed before completion.


## Supplementary encounter kinematics

Speeds are estimated from a centered quadratic fit over five native observations (40 ms). Angles are relative to the fixed positive x direction; speed gain is the difference between tip and attachment x velocity. These are offline diagnostics, not impact measurements.

| Take | Tip speed at planned time (m/s) | Direction angle to +x (deg) | Forward speed gain over root (m/s) |
|---|---:|---:|---:|
| M0_001 | 6.20 | 35.3 | 5.98 |
| M0_002 | 5.97 | 34.7 | 5.83 |
| M0_003 | 6.60 | 36.6 | 6.26 |
| M0_004 | 5.15 | 4.9 | 6.01 |
| M0_005 | 5.98 | 10.2 | 6.83 |

[LaTeX results table](M0_results_table.tex). This table is not inserted into the manuscript automatically.
