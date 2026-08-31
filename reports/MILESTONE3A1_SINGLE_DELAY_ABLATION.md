# Milestone 3A.1 — Single-Delay UAV Model Ablation

Stage: **UAV command-delay ablation only**  
Stage B EI/Cb: **LOCKED / NOT RUN**  
Stage C joint fitting: **LOCKED / NOT RUN**

## Outcome

The accepted five-parameter Milestone 3A model and artifact were preserved. A separate six-parameter candidate was fitted with one effective command-input latency, `tau_cmd`, bounded to 0–80 ms. Commands were evaluated causally at `t - tau_cmd` using zero-order hold; measurements and state-initialization timestamps were not shifted.

**Model decision: Retain the accepted five-parameter baseline.**

## Baseline provenance

- Accepted baseline artifact: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T035156.958739+0000_a0319d7f`
- Delay artifact: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T121702.805149+0000_66677ae3`
- Baseline authoritative training: 42.8 mm / 5.44 deg.
- Baseline authoritative validation: 61.2 mm / 8.24 deg.
- All accepted baseline windows had sufficient 80-ms command history, so the direct comparison uses exactly the original 321 training and 188 validation windows.

## Model and fitted parameters

The accepted dynamics are unchanged. The candidate only replaces each command input `u(t)` with the last recorded command whose timestamp is at or before `t - tau_cmd`.

| Parameter | 5-param baseline | 6-param + delay |
|---|---:|---:|
| K_p | 24.630215 | 19.673339 |
| K_v | 24.363403 | 11.118280 |
| k_a | 1.051226 | 1.925217 |
| K_R | 59.591014 | 73.915840 |
| K_omega | 8.910099 | 9.124682 |
| tau_cmd [s] | 0 (no delay parameter) | 0.080 |

`tau_cmd` bound: [0.0, 0.08] s. Bound hit: **YES**. Profile resolution: 0.010 s. The delay is an aggregate effective latency and is not assigned to a specific hardware component.

## Fair common-window comparison

| Split | windows | baseline position [mm] | delayed position [mm] | baseline orientation [deg] | delayed orientation [deg] |
|---|---:|---:|---:|---:|---:|
| Training | 321 | 42.8 | 30.5 | 5.44 | 5.21 |
| Validation | 188 | 61.2 | 44.6 | 8.24 | 7.43 |

| Take | role | windows | baseline position [mm] | delayed position [mm] | baseline orientation [deg] | delayed orientation [deg] |
|---|---|---:|---:|---:|---:|---:|
| fig8_001 | training | 158 | 21.6 | 22.7 | 4.52 | 4.55 |
| fig8_002 | training | 99 | 33.9 | 29.5 | 6.05 | 5.70 |
| fig8_003 | validation | 188 | 61.2 | 44.6 | 8.24 | 7.43 |
| osc_001 | training | 64 | 62.3 | 37.4 | 5.63 | 5.32 |

## Held-out lead-time comparison

| lead [s] | baseline position [m] | delayed position [m] | baseline orientation [deg] | delayed orientation [deg] |
|---:|---:|---:|---:|---:|
| 0.1 | 0.0122 | 0.0069 | 3.69 | 3.09 |
| 0.25 | 0.0350 | 0.0247 | 7.21 | 6.78 |
| 0.5 | 0.0632 | 0.0472 | 9.23 | 8.35 |
| 1.0 | 0.0847 | 0.0609 | 9.46 | 8.35 |

## Axis and Euler diagnostics

Each cell is `baseline / delayed`.

| Take | X [mm] | Y [mm] | Z [mm] | roll [deg] | pitch [deg] | yaw [deg] |
|---|---:|---:|---:|---:|---:|---:|
| fig8_001 | 12.7 / 9.0 | 9.9 / 4.9 | 14.4 / 20.2 | 1.23 / 1.01 | 4.32 / 4.42 | 0.48 / 0.46 |
| fig8_002 | 24.4 / 14.2 | 18.3 / 15.8 | 14.7 / 20.5 | 4.01 / 3.40 | 4.48 / 4.51 | 1.10 / 1.04 |
| fig8_003 | 46.0 / 22.8 | 34.5 / 24.9 | 21.1 / 29.1 | 6.55 / 5.47 | 4.96 / 4.90 | 2.50 / 2.29 |
| osc_001 | 56.4 / 18.8 | 15.5 / 14.1 | 21.3 / 29.1 | 1.04 / 1.07 | 5.52 / 5.19 | 0.44 / 0.43 |

## Residual timing

Positive lag means the measured response occurs later than the prediction.

| Take | baseline best translation lag [s] | delayed best translation lag [s] |
|---|---:|---:|
| fig8_001 | 0.030 | 0.010 |
| fig8_002 | 0.020 | 0.000 |
| fig8_003 | 0.020 | 0.000 |
| osc_001 | 0.040 | 0.010 |

## Delay sensitivity

Because the processed command is zero-order held, the objective is piecewise constant in delay. The bounded delay was therefore profiled at the recorded 10-ms grid rather than using forbidden linear interpolation. This identifies a delay bin, not sub-frame confidence.

| tau [s] | final training objective |
|---:|---:|
| 0.000 | 1.524011 |
| 0.010 | 1.303197 |
| 0.020 | 1.112521 |
| 0.030 | 0.950194 |
| 0.040 | 0.818020 |
| 0.050 | 0.714346 |
| 0.060 | 0.639342 |
| 0.070 | 0.594147 |
| 0.080 | 0.577964 |

## Model-selection gates

| Gate | Result |
|---|---|
| tau not on bound | FAIL |
| heldout position materially improved at least 10 percent | PASS |
| more than one training motion improved | PASS |
| heldout orientation not degraded more than 5 percent | PASS |
| residual translation timing reduced | PASS |

- Held-out position relative improvement: 27.1%.
- Held-out orientation ratio, delay/baseline: 0.901.
- Training takes with lower position RMSE: ['fig8_002', 'osc_001'].
- Decision: **RETAIN_FIVE_PARAMETER_BASELINE**.
- The 10% held-out position and 5% orientation non-degradation thresholds are report-level materiality gates, not fit tuning criteria.

## Window-history audit

| Take | original accepted | delay eligible | rejected for tau_max history |
|---|---:|---:|---:|
| fig8_001 | 158 | 158 | 0 |
| fig8_002 | 99 | 99 | 0 |
| fig8_003 | 188 | 188 | 0 |
| osc_001 | 64 | 64 | 0 |

Every eligible window has valid command history through `t_start - 0.080 s`. Rejections and exact window boundaries are stored in `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T121702.805149+0000_66677ae3\delay_window_audit.json`. Initial measured UAV state remains at the physical window start.

## Optimization and reproducibility

- Sobol candidates: 32 over the same five gain bounds, profiled over every tau bin.
- Adam iterations: 100 on the five differentiable gains.
- Tau was re-profiled every 10 Adam iterations and once at the final gain state.
- Hierarchical weighting, loss normalization, train/validation take roles, causal initialization, and lead-time metrics are unchanged.
- Runtime: 482.8 s on `cuda` using `torch.float64`.
- Dataset snapshot: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T121702.805149+0000_66677ae3\dataset_snapshot.json`.
- Optimization trace: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results\2026-08-28T121702.805149+0000_66677ae3\optimization_trace.csv`.

## Scientific stop

No EI/Cb fitting, cable loss, joint refinement, axis-specific gain, second delay, drag, residual model, or measurement-time shift was introduced. Stage B and Stage C remain locked pending review of this model comparison.
