# Figure-8 endpoint-history observer generalization

## Executive result

The frozen four-mode endpoint-history observer generalizes beyond the original favorable `sin(pi*s)` disturbance **for this matched-physics Figure-8 task**. Across the primary disturbances, it recovered **42–213%** of the finite-seed endpoint/full-state tracking gap whenever the endpoint baseline was worse than full state. More importantly, it reduced 0.2 s future-tip prediction error from **83.5 to 14.7 mm** for `sin(pi*s)`, from **87.6 to 4.0 mm** for out-of-basis `sin(4pi*s)`, and from **115.0 to 11.5 mm** for the mixed represented/unrepresented disturbance.

The central finding is not exact recovery of every injected mode. `sin(4pi*s)` and `sin(5pi*s)` have basis residuals 0.959 and 0.979, yet their history-observer 0.2 s prediction errors are only 4.0 and 3.3 mm. Their high spatial-frequency momentum decays quickly under the matched damped DDER, and the fixed basis corrects the lower-dimensional state that remains relevant to the future tip. The evidence therefore supports interpreting this estimator as a **control-relevant distributed-state observer**, not an exact inverse of arbitrary cable state.

## Frozen method and scope

The observer and controller mathematics were not retuned. The observer used a 0.30 s causal history; the nominal 50 Hz history contains 16 frames and 15 transitions, while actual timestamp retention occasionally produces 17 frames. Its four spatial functions were `s`, `sin(pi*s)`, `sin(2pi*s)`, and `sin(3pi*s)`; 24 position/velocity coefficients; 0.025 m and 0.25 m/s correction scales; prior and LM weights `1e-4`; central finite-difference step 0.04; the existing bounds, four-value line search, and two GN/LM iterations.

The plant, observer, and controller used identical EI and Cb. There was no adaptation, sensing noise, delay, dropout, or future measurement. Only plant cable velocity was disturbed. All non-clean disturbances were normalized to the RMS of `0.45 sin(pi*s)`, which is 0.3182 m/s over the ten dynamic nodes; root and tip position and velocity were unchanged at injection. The random smooth mixture used seed 20260826 and its coefficients are preserved in the machine-readable metadata.

The current production GUI configuration—not the older 2,048-sample study setting—was authoritative: horizon 1.0 s, 1024 candidates, 2 MPPI iterations, 11 acceleration knots, 50 Hz physics/control, 10 Hz replanning, and 10 Hz observer correction. This explains why numerical values are a new paired experiment rather than an exact replay of the older 2,048-sample report.

## Sequential online timing

The timing sanity check used 37 warmed active updates from one run, with the observer followed by MPPI and an explicit CUDA synchronization at command availability.

| component | mean | median | p95 | maximum |
|---|---:|---:|---:|---:|
| observer | 21.72 ms | 22.16 ms | 24.37 ms | 24.73 ms |
| MPPI | 95.69 ms | 95.63 ms | 99.44 ms | 101.87 ms |
| sequential total | 117.71 ms | 117.93 ms | 122.84 ms | 124.79 ms |
| unattributed | 0.287 ms | 0.246 ms | 0.490 ms | 0.837 ms |

Mean sequential capacity is 8.50 Hz and the p95-time capacity is 8.14 Hz. Thus the current 10 Hz request is not met sequentially; this is a measured scheduling limitation, not an observer-method change. Prewarming both 16- and 17-frame static history graphs removed an avoidable first-use graph-capture outlier without changing estimation.

## Main primary-disturbance results

Values are pooled RMS across paired MPPI seeds 17, 23, and 41; Jacobian diagnostics are medians. Full state has zero estimator error by construction. Tracking-gap recovery is intentionally unclamped.

| disturbance | basis residual | rank | condition | tracking full / endpoint / history (mm) | gap recovered | state pos / vel RMSE | tip prediction 0.2 s: endpoint / propagation / history (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.000 | 18/24 | 3.51e+05 | 17.5 / 28.1 / 17.9 | 96.4% | 0.0 mm / 0.000 m/s | 70.7 / 0.0 / 0.0 |
| sin(πs) | 0.000 | 18/24 | 3.17e+05 | 32.1 / 42.1 / 37.9 | 42.1% | 9.2 mm / 0.059 m/s | 83.5 / 39.4 / 14.7 |
| sin(2πs) | 0.000 | 18/24 | 2.45e+05 | 22.0 / 37.1 / 23.7 | 88.7% | 5.0 mm / 0.043 m/s | 75.1 / 21.4 / 4.7 |
| sin(3πs) | 0.000 | 19/24 | 2.38e+05 | 20.1 / 37.9 / 22.7 | 85.3% | 3.8 mm / 0.034 m/s | 92.4 / 12.3 / 6.1 |
| sin(4πs) | 0.959 | 18/24 | 3.68e+05 | 21.6 / 25.3 / 17.4 | 212.7% | 2.7 mm / 0.028 m/s | 87.6 / 9.5 / 4.0 |
| sin(5πs) | 0.979 | 18/24 | 3.71e+05 | 18.1 / 35.3 / 17.8 | 101.8% | 2.1 mm / 0.024 m/s | 98.1 / 6.4 / 3.3 |
| represented mix | 0.000 | 18/24 | 3.30e+05 | 36.1 / 57.1 / 40.9 | 77.3% | 9.1 mm / 0.055 m/s | 89.1 / 42.9 / 8.4 |
| mixed span | 0.454 | 19/24 | 2.84e+05 | 28.1 / 51.1 / 33.6 | 76.2% | 7.8 mm / 0.051 m/s | 115.0 / 33.0 / 11.5 |
| random smooth | 0.397 | 18/24 | 3.25e+05 | 25.3 / 32.0 / 24.9 | 106.0% | 5.9 mm / 0.044 m/s | 62.8 / 20.7 / 11.7 |

The clean test is important: the observer state error remains numerically zero and tracking is essentially full-state quality (17.9 versus 17.5 mm). The observer therefore does not invent a material correction when recursive matched DDER already explains the history.

Represented modes are recoverable. `sin(2pi*s)` and `sin(3pi*s)` attain history position RMSE 5.0 and 3.8 mm, and recover 89% and 85% of the control gap. The represented mixture is harder: its state velocity RMSE is 0.055 m/s and tracking recovery is 77%, but its 0.2 s tip prediction is still 8.4 mm versus 89.1 mm for instantaneous endpoint reconstruction.

Basis residual is not a monotonic predictor of state or control error in this task. The unrepresented high modes have large instantaneous projection residual, but propagation alone already predicts them relatively well after their fast physical decay. Endpoint correction further improves that prediction. The mixed-span and random disturbances retain residuals 0.454 and 0.397; their history state position errors are 7.8 and 5.9 mm, while 0.2 s tip errors remain 11.5 and 11.7 mm. This is direct evidence that full-state reconstruction error and control-relevant prediction error are distinct.

## Observability and optimization behavior

The median numerical rank is 18–19 of 24 for every primary case, with condition numbers approximately `2e5–4e5`. Endpoint history therefore does not independently constrain all 24 correction coordinates. This is a structurally ill-conditioned local inverse problem, stabilized by the frozen prior, LM damping, trust region, and physical rollout. The reported rank should not be interpreted as proof that each injected mode is uniquely identified.

For every non-clean primary run, all 17 ready corrections were accepted; for every clean run, all 17 were correctly rejected. Accepted corrections used both allowed GN/LM iterations. All 816 accepted line-search steps selected `alpha=1.0`, and there were zero trust-region or component-bound saturations. Mean correction norm ranged from 0.047 for `sin(5pi*s)` to 0.139 for `sin(pi*s)`. Residual reduction was systematic: for example, `sin(pi*s)` fell from 4.35 to 1.76 mm, the represented mixture from 3.61 to 0.95 mm, and the mixed-span case from 3.64 to 1.49 mm. This indicates stable local optimization in the tested basin; it does not resolve the rank deficiency. No case-specific parameter was changed after examining these diagnostics. Observer timing stayed close to the independently measured 21.72 ms mean; disturbance shape did not create a material timing change because all captured workloads have the same shape.

Propagation-only is a necessary control. It is already strong for high modes because the correct DDER dissipates their unobserved momentum, but endpoint-history correction improves 0.2 s prediction for every disturbed primary case. For example, propagation/history errors are 39.4/14.7 mm for `sin(pi*s)`, 9.5/4.0 mm for `sin(4pi*s)`, and 33.0/11.5 mm for the mixed case. Endpoint history therefore adds information beyond perfect-model propagation from the wrong prior.

## Direction generalization

| case | full / endpoint / history tracking RMSE (mm) | gap recovered | history state position RMSE (mm) |
|---|---:|---:|---:|
| mixed span x | 36.1 / 54.1 / 39.6 | 80.8% | 7.9 |
| mixed span xy | 19.4 / 29.5 / 16.7 | 126.5% | 7.8 |
| sin1 x | 41.4 / 54.1 / 47.2 | 54.6% | 9.4 |
| sin1 xy | 20.6 / 22.8 / 21.4 | 64.8% | 9.3 |
| sin4 x | 19.2 / 31.5 / 23.9 | 61.9% | 2.7 |
| sin4 xy | 13.9 / 24.4 / 17.9 | 61.8% | 2.7 |

The observer benefit is not specific to the original y direction. Every selected x and normalized xy case moves toward full-state tracking relative to instantaneous endpoint reconstruction; tracking-gap recovery ranges from roughly 55% to 126%. No z disturbance was tested, as specified.

## Mid-run recovery

At t=0.8 s the plant alone received the hidden velocity impulse. The controller and observer were not told its type, coefficients, direction, magnitude, or time. Pooled tracking-gap recovery was 84% for `sin(pi*s)`, 93% for `sin(4pi*s)`, and 114% for the mixed-span disturbance. The time histories show the expected causal sequence: the observer is wrong immediately at injection, the tip signature develops, corrections are accepted at later 10 Hz updates, distributed error falls, and tracking approaches the full-state case. There is no instantaneous acausal recovery.

## Figures

1. [Tracking RMSE](figure8_history_observer_generalization_data/figures/01_tracking_rmse.png)
2. [Tracking-gap recovery](figure8_history_observer_generalization_data/figures/02_tracking_gap_recovery.png)
3. [Distributed position error](figure8_history_observer_generalization_data/figures/03_position_rmse.png)
4. [Distributed velocity error](figure8_history_observer_generalization_data/figures/04_velocity_rmse.png)
5. [Future-tip prediction](figure8_history_observer_generalization_data/figures/05_future_tip_prediction.png)
6. [Basis residual relationships](figure8_history_observer_generalization_data/figures/06_07_basis_relationships.png)
7. [Jacobian rank and condition](figure8_history_observer_generalization_data/figures/08_09_observability.png)
8. [History residual before/after correction](figure8_history_observer_generalization_data/figures/10_history_residual.png)
9. [Initial-disturbance recovery](figure8_history_observer_generalization_data/figures/11_initial_recovery_time.png)
10. [Mid-run recovery](figure8_history_observer_generalization_data/figures/12_midrun_recovery_time.png)
11. [Cable-state snapshots](figure8_history_observer_generalization_data/figures/13_cable_snapshots.png)
12. [Direction generalization](figure8_history_observer_generalization_data/figures/14_direction_generalization.png)

## Answers to the study questions

1. **`sin(2pi*s)` and `sin(3pi*s)`:** yes. Both are reconstructed and both approach full-state prediction/control.
2. **Represented mixture:** yes, though it is harder than either higher single represented mode.
3. **`sin(4pi*s)` and `sin(5pi*s)`:** they are not representable at injection, but they decay rapidly and the observer recovers the lower-frequency control-relevant remainder. They are not failures for this task.
4. **Representability versus state accuracy:** weak, non-monotonic relationship in this experiment. Fast DDER dynamics matter as much as static projection error.
5. **Representability versus tip prediction:** also weak. Large basis residual can coexist with 3–4 mm prediction error at 0.2 s.
6. **Poor full-state versus accurate tip prediction:** yes; the represented mixture and mixed-span cases show this separation most clearly.
7. **Imperfect state versus near-full control:** yes. The random smooth and high-mode cases reach or exceed finite-seed full-state tracking despite nonzero state error.
8. **Visible modes:** all tested disturbances produce enough endpoint signature for useful correction over 0.30 s, but visibility is only local and rank deficient.
9. **Poorly observable modes:** no primary disturbance is a control failure, but correction coordinates remain ill-conditioned and individual coefficients are not uniquely observable.
10. **History versus propagation:** history improves future-tip prediction for every disturbed primary case.
11. **History versus instantaneous endpoint:** history improves future-tip prediction and pooled tracking in every primary disturbance; the amount varies substantially.
12. **Mid-run disturbances:** yes, all three selected cases recover causally after t=0.8 s.
13. **Adequacy of four-mode basis:** adequate for control-relevant hidden dynamics in this matched, damped, flat Figure-8 task. It is not an exact universal cable-state basis.
14. **Dominant limitation:** endpoint-history observability/conditioning, followed by static basis expressiveness. Optimization converged consistently enough that it is not the dominant observed failure, and high-mode control relevance is low because those modes decay quickly.
15. **Proceed to imperfect EI/Cb:** yes, but freeze this state observer and vary only model parameters in the next study.

## Limitations

This is three paired seeds, matched physics, exact root/tip observations, no timing jitter, and one Figure-8 geometry. MPPI is stochastic, so recovery above 100% is finite-sample behavior, not evidence that partial observation is intrinsically superior to full state. High-mode success depends on the present damping and task horizon; it does not prove arbitrary hidden modes are harmless in other cables or tasks. The observer Jacobian is rank deficient, so state correction should not be described as unique physical mode identification.

## Recommendation

**Freeze the compact history observer as a control-relevant state estimator and proceed to an imperfect-EI/Cb study.** Do not enlarge the basis yet. The current basis gives accurate future-tip prediction and near-full-state control even when exact high-mode reconstruction is impossible; the next isolated uncertainty should be model mismatch, not additional observer complexity.

## Reproducibility

Machine-readable summaries, per-seed trajectories, observer diagnostics, timing, tables, and plots are in `reports/figure8_history_observer_generalization_data/`. The experiment runner is `research_tools/figure8_history_observer_generalization_study.py`; this report generator is `research_tools/report_figure8_history_observer_generalization.py`.
