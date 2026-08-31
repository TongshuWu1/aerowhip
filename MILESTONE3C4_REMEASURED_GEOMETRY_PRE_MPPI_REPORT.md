# Milestone 3C.4 — Remeasured-Geometry Cable Refit and PRE_MPPI Report

**Repository:** `C:\Users\wts28\Documents\PHD\particle_filter_cable_project`  
**Status:** **PASS**  
**Fit artifact:** `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results_remeasured_geometry\2026-08-29T020130.426074+0000_f9fcdad8`

## 1. Geometry and reference audit

The audit passed. Motive rigid body `cf_7` is copied into `uav_position_m/uav_orientation_xyzw` with an identity source-to-simulator transform. The production clamp then applies `p_connector = p_rigid_body + R(q)d_body`. No alternate pose origin or hidden translation was found.

- Attachment offset: `[0.0, 0.0, -0.055]` m (55 mm body -Z).
- Cable length, connector to c10: **952.5 mm**.
- Rest lengths [m]: `[0.0315, 0.0315, 0.087, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1025, 0.1, 0.1]`.
- Topology: 12 nodes, 11 edges; node 0 and node 1 prescribed; c1...c10 map to nodes `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`.
- Connector-to-c1 is 63 mm, split into two 31.5-mm clamp edges. Node 1 is not an observation.

## 2. Mass audit

- Bare cable: old/new = 0.00700000 / 0.00700000 kg.
- Marker total: old/new = 0.00909091 / 0.00909091 kg.
- Total modeled: old/new = 0.01609091 / 0.01609091 kg.
- The edge-dual-volume bare masses were recomputed from the new rest lengths; old node masses were not copied.

## 3. Frozen upstream model and backend

- UAV Physics was **not refitted**.
- UAV Residual was **not retrained**.
- Physics-only Model P was **not evaluated**.
- Production path: CUDA float32, PCG32, fused DDER/projection, 3 substeps, 4 position projections.

## 4. EI/Cb search

The fit used all accepted Training PhysicalEpisodes, each initialized once at its original causal start and propagated continuously to its original end. The loss used all valid c1...c10 observations, the frozen pseudo-Huber scale, and equal top-level take weighting.

| Pass | Best EI [N m²] | Best Cb [N m² s] | Objective | Runtime [s] |
|---:|---:|---:|---:|---:|
| 1 | 0.0002 | 0.0015 | 9.18662117e-05 | 85.47 |
| 2 | 1.96314227e-05 | 0.0030687495 | 8.13133302e-05 | 84.30 |
| 3 | 9.22579758e-05 | 0.002770457 | 8.06249227e-05 | 85.00 |

- Final EI: **9.22579757946e-05 N m²** — **contained**.
- Final Cb: **0.00277045699567 N m² s** — **weak**.
- Best objective: 8.06249227026e-05.
- EI adjacent profile evidence: `[{'EI': 6.266007128797605e-05, 'profile_objective': 8.216655987780541e-05, 'relative_rise_from_minimum': 0.019121099575354402}, {'EI': 0.00013583664880629958, 'profile_objective': 8.293376595247537e-05, 'relative_rise_from_minimum': 0.028636842957836407}]`.
- Cb adjacent profile evidence: `[{'Cb': 0.0022580386007495755, 'profile_objective': 8.289262041216716e-05, 'relative_rise_from_minimum': 0.02812651018567359}, {'Cb': 0.003068749501516041, 'profile_objective': 8.131333015626296e-05, 'relative_rise_from_minimum': 0.008538395208737835}]`.
- EI full profiled span: 33.771%.
- Cb full profiled span: 452.084%.

## 5. Conditional task-horizon validation

Measured Motive UAV pose drove the production rigid clamp. Each eligible provisional-validation PhysicalEpisode was initialized once and propagated as one continuous prefix for at most 1.0 s; there were no rolling windows or resets.

At 0.70 s:

- Distributed c1...c10 RMSE: **44.101 mm**.
- c10 tip RMSE: **71.543 mm**.

Full lead-time data are in `conditional_task_horizon_validation.json`.

## 6. PR-only end-to-end task-horizon validation

The only end-to-end predictor evaluated was recorded FullState → frozen UAV Physics + frozen UAV Residual → rigid clamp → fitted 12-node DDER. No measured UAV or cable state entered after the one causal initialization.

Lead-time aggregation first computes RMSE across eligible episodes within each physical take and then gives `fig8_003` and `osc_003` equal top-level weight. Final verification corrected an initially episode-weighted summary before the freeze was finalized; no trajectory, fitted parameter, or per-episode metric changed.

At 0.70 s:

- UAV position RMSE: **26.967 mm**.
- UAV orientation RMSE: **4.014 deg**.
- Distributed c1...c10 RMSE: **55.370 mm**.
- c10 tip RMSE: **82.555 mm**.

## 7. Saved historical comparison

No old-geometry trajectory was rerun.

| Metric at 0.70 s | Old 0.943-m saved | New 0.9525-m |
|---|---:|---:|
| Conditional distributed | 42.108 mm | 44.101 mm |
| Conditional tip | 66.287 mm | 71.543 mm |
| PR UAV position | 11.491 mm | 26.967 mm |
| PR UAV orientation | 4.112 deg | 4.014 deg |
| PR distributed cable | 53.286 mm | 55.370 mm |
| PR tip | 103.716 mm | 82.555 mm |

## 8. Frozen acceptance decision

The predeclared 0.70-s checks were: `{'distributed_marker_rmse_m': True, 'tip_rmse_m': True, 'uav_orientation_rmse_deg': True, 'uav_position_rmse_m': True}`. Fatal rollout failures: `[]`. Initialization-ineligible episodes remain explicit exclusions: `[]`.

**PRE_MPPI = PASS.** `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI` was created and selected explicitly.

## 9. Production status and protected test

- Active-model status after the run: `MODEL_FROZEN_FOR_MPPI`.
- Identification UI source: the explicit active-model/result status API; no newest-timestamp selection.
- `fig8vertical_002`: **PROTECTED — NOT EVALUATED**.

## 10. Runtime and reproducibility

- Total workflow runtime: 267.32 s.
- Git commit: `cbdb59b4f05472506599b6f7e08be08e05541ff3`.
- Aggregate source SHA-256: `56ab1d50590f668f8aaf29a8390793b320d46ebb8cbcc60b1627acad47e811df`.
- Working tree dirty state was recorded: `True`.
- Detailed source/config hashes: `source_hash_manifest.json`.

## 11. Final decision

    EI = 9.22579757946e-05
    status = contained

    Cb = 0.00277045699567
    status = weak

    Conditional @ 0.70 s:
        distributed = 44.101 mm
        tip = 71.543 mm

    PR End-to-End @ 0.70 s:
        UAV position = 26.967 mm
        UAV orientation = 4.014 deg
        distributed cable = 55.370 mm
        tip = 82.555 mm

    PRE_MPPI = PASS

Generic system identification is now frozen. The next scientific milestone is task formulation plus MPPI.
