# Milestone 3C.1 — 12-Node Production Cable Report

Generated: `2026-08-28T22:16:22.975549+00:00`  
Methodology: `milestone3c1_12node_measured_boundary_grid_v1`

## 1. Outcome

- Parameter identification: **C — EI/Cb still boundary seeking**.
- Task-horizon model: **PASS** around 0.7 s. This classification requires finite predictions, no coupled-rollout failure, and the frozen usability thresholds `{'uav_position_rmse_m': 0.05, 'uav_orientation_rmse_deg': 20.0, 'distributed_marker_rmse_m': 0.15, 'tip_rmse_m': 0.125}` across cable-eligible provisional-validation episodes. Initialization-ineligible episodes remain explicit exclusions rather than numerical failures.
- Protected `fig8vertical_002`: **not evaluated**.

## 2. Code and configuration changed

- `config/default.json`: production interval subdivisions changed from ten two-edge intervals to `[2,1,1,1,1,1,1,1,1,1]`.
- `simulator/cable/config.py`: supports per-measured-interval subdivisions and derives node mapping, rest lengths, and lumped masses.
- `fitting/initialization.py`: latent nodes are interpolated from material coordinates instead of odd-node assumptions.
- `simulator/observation/optitrack.py`: c1...c10 map to nodes 2...11.
- `simulator/cable/cuda_fixed_pcg.py`, `simulator/cable/dder.py`, `simulator/simulator.py`: the unchanged generated PCG32/fused mechanics accepts 12 nodes.
- `fitting/decomposed.py`: existing evaluators additionally report distributed-marker error at requested lead times.
- `fitting/milestone3c1.py` and `run_milestone3c1.py`: bounded forward-only grid orchestration and frozen-model validation.

No bending, damping, projection, precision, substep, root-attachment, UAV, or residual equation was changed.

## 3. Topology and observation contract

| Model | Clamp nodes | Cable marker nodes |
|---|---|---|
| Saved 21-node | `[0,1]` | `[2,4,6,8,10,12,14,16,18,20]` |
| Production 12-node | `[0,1]` | `[2,3,4,5,6,7,8,9,10,11]` |

- Cable observations: exactly `c1...c10` (10 points).
- UAV/root and clamp-support nodes are absent from cable loss and RMSE.
- c10 remains dynamic node 11 and the free tip.
- Rigid boundary remains `r0=p+R*d`, `r1=r0+l0*normalize(R*t)`.

Rest lengths [m]: `[0.03, 0.03, 0.085, 0.1, 0.1, 0.098, 0.1, 0.1, 0.1, 0.1, 0.1]`  
Sum: `0.943 m`  
Authoritative physical length: `0.943 m`

## 4. Mass audit

| Quantity | 21-node | 12-node |
|---|---:|---:|
| Bare cable [kg] | 0.007 | 0.007 |
| Marker mass [kg] | 0.00909091 | 0.00909091 |
| Total modeled [kg] | 0.01609091 | 0.01609091 |

Mass is redistributed by the existing edge-dual-volume lumping rule; old node masses were not copied.

## 5. Scientific identification problem

Current production fitting retains a rigid two-node clamp, complete PhysicalEpisode propagation, one initialization per episode, and no cable reset. The canonical legacy fitter prescribed only node-0 position, left the tangent free, used approximately one-second windows, and repeatedly reinitialized the distributed cable. Historical EI/Cb values are therefore not reconciled here.

## 6. EI/Cb grid search

| Pass | Kind | EI range | Cb range | Best EI | Best Cb | Objective | Runtime [s] |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | broad_initial | [1e-08, 0.004] | [1e-10, 1.5e-05] | 0.000633527844 | 1.5e-05 | 0.000134470509 | 87.25 |
| 2 | boundary_extension | [0.00010033938212454077, 0.004000000000000001] | [2.73306304948162e-06, 0.00015] | 0.000824315492 | 0.00015 | 0.000122863101 | 85.56 |


Final:

```text
EI = 0.000824315491667 N m^2
Cb = 0.00015 N m^2 s
```

- EI profile: `{'status': 'contained', 'minimum_index': 4, 'minimum_objective': 0.00012286310084164143, 'relative_adjacent_profile_rise': [0.04674770652633255, 0.012675457131324463], 'relative_full_profile_span': 0.21424767810071385, 'weak_threshold_relative_adjacent_rise': 0.001}`.
- Cb profile: `{'status': 'boundary_seeking', 'minimum_index': 7, 'minimum_objective': 0.00012286310084164143, 'relative_adjacent_profile_rise': [0.032417759947545255], 'relative_full_profile_span': 0.08968661702921964, 'weak_threshold_relative_adjacent_rise': 0.001}`.
- Training objective: `0.000122863100842`.
- The saved CSV grids, `ei_cb_profile_curves.png`, and `final_ei_cb_loss_landscape.png` are in `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results_12node\2026-08-28T221032.685038+0000_0eea8215`.

## 7. Conditional measured-boundary validation

- Distributed c1...c10 RMSE: `214.184 mm`.
- c10 tip RMSE: `363.193 mm`.
- Terminal c10 RMSE: `147.803 mm`.

| Lead [s] | distributed_marker_rmse_m | tip_rmse_m |
|---:|---:|---:|
| 0.1 | 8.227 | 4.360 |
| 0.25 | 17.124 | 11.975 |
| 0.5 | 30.149 | 44.130 |
| 0.7 | 42.108 | 66.287 |
| 1.0 | 37.545 | 57.109 |

All distance columns above are millimetres.

## 8. End-to-end frozen UAV physics + residual + 12-node DDER

| Lead [s] | uav_position_rmse_m | uav_orientation_rmse_deg | distributed_marker_rmse_m | tip_rmse_m |
|---:|---:|---:|---:|---:|
| 0.1 | 4.225 | 3.390 | 9.523 | 7.556 |
| 0.25 | 11.785 | 5.284 | 23.320 | 8.480 |
| 0.5 | 9.930 | 4.084 | 39.036 | 53.019 |
| 0.7 | 11.491 | 4.112 | 53.286 | 103.716 |
| 1.0 | 13.556 | 12.541 | 54.927 | 73.343 |

Distance columns are millimetres; orientation is degrees. No measured UAV or cable state enters after the single causal initialization. Failed/ineligible episodes: `[{'take_id': 'fig8_003', 'view_id': 'fig8_003__episode_001__residual_suffix', 'reason': 'cable_initialization_ineligible', 'detail': 'initialization_rmse=0.0118034 exceeds 0.01 m'}]`.

At 0.7 s the threshold checks were `{'uav_position_rmse_m': True, 'uav_orientation_rmse_deg': True, 'distributed_marker_rmse_m': True, 'tip_rmse_m': True}`. Fatal rollout failures were `[]`.

## 9. Saved 21-node comparison

| Quantity | Saved 21-node | 12-node |
|---|---:|---:|
| EI [N m²] | 0.0004 | 0.000824315492 |
| Cb [N m² s] | 7.82323594e-07 | 0.00015 |
| Training objective | 0.000130944929 | 0.000122863101 |
| Conditional validation markers [mm] | 222.196 | 214.184 |
| Conditional validation tip [mm] | 378.782 | 363.193 |
| Cable-fit runtime [s] | 272.91 | 172.82 |

The 21-node fit used three grid passes. The 12-node runtime covers only the passes required by the frozen adaptive rule, so runtime is reported rather than treated as a controlled benchmark.

Saved 21-node versus 12-node task-horizon values (the old artifact did not contain distributed-marker lead-time errors or 0.7-s values):

| Lead [s] | Model | UAV position [mm] | UAV orientation [deg] | Tip [mm] |
|---:|---|---:|---:|---:|
| 0.5 | saved 21-node | 9.930 | 4.084 | 52.570 |
| 0.5 | 12-node | 9.930 | 4.084 | 53.019 |
| 0.7 | saved 21-node | unavailable | unavailable | unavailable |
| 0.7 | 12-node | 11.491 | 4.112 | 103.716 |
| 1.0 | saved 21-node | 13.556 | 12.541 | 65.519 |
| 1.0 | 12-node | 13.556 | 12.541 | 73.343 |

## 10. Runtime

- EI/Cb grid total: `172.82 s`.
- Conditional validation: `112.75 s`.
- End-to-end validation: `61.82 s`.
- Total milestone execution: `350.29 s`.

## 11. Reproducibility and invariants

- Git: `{'commit': 'cbdb59b4f05472506599b6f7e08be08e05541ff3', 'working_tree_dirty': True, 'working_tree_status_sha256': '5fa6aef2b645a6bc08ab72e24123dfd22734409810773034c3af0781e6e1157f'}`.
- Source hashes: `{'config/default.json': '54f8ca635b80a572949bd850d7dd744ae9faf330931fa0d75b4b6e59a35f530c', 'fitting/decomposed.py': 'f5b13195afb7e953282b134d49859250212e6d48a56afa20c726720853a0615b', 'fitting/initialization.py': 'af490f05de2d0319c6630ad77bcc6007fe30451ea2fc66bff6631644eae8e4c2', 'fitting/milestone3c1.py': '376d548c0af0e365b9b176f46d876f27e497741081cfaf26804c0fe4fa54ec55', 'simulator/cable/config.py': 'a0162d7db71213bc45773c6164096114321a73b7168f7e107ffeec324745cb61', 'simulator/cable/cuda_fixed_pcg.py': '4cf85d89b0f01922207880b3e64d150194f565788b74fefbba6e916b1f6ee635', 'simulator/cable/dder.py': '94b12120d7e2358bda428159103bed658ae08916ac910c01ec53da823892f87f', 'simulator/coupling/attachment.py': '2ec33209eacbd551f01605eec6e854b8d45a54dade6bb3fccfae9db3a5591c75', 'simulator/coupling/root_boundary.py': '7b326089a49585f18c323d03501d12352fa154a2c91b318d3f7ef18e6b215672', 'simulator/observation/optitrack.py': 'ea9ea7454d05da52b7b46413dc50329c216d1c6d18fc242351c85f159a229512', 'simulator/simulator.py': '4c45da52d7ce3f0c1cfbbbd7e71c786f8640121539ca8dfa66ba97b79e580dc4'}`.
- Same isotropic EI/Cb DDER equations: yes.
- Same three substeps/four position projections: yes.
- Same float32 no-grad production PCG32 backend: yes.
- Same pseudo-Huber scale and marker masks: yes.
- UAV parameters and residual: loaded unchanged from `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\model_freezes\MODEL_FREEZE_DECOMPOSED_PRETEST`.
- Protected test evaluated: no.

## 12. Scientific conclusion

**Parameter identification: C — EI/Cb still boundary seeking.** EI is contained at `8.24315491667e-4 N m²`: the neighboring profiled objectives rise by 4.67% and 1.27%. Cb is not contained: the loss continues decreasing to `1.5e-4 N m² s`, the upper edge of the one permitted Cb extension. The search therefore stopped as required. The selected Cb is the best bounded production candidate from this milestone, but it must not be presented as a well-identified material value.

**Task-horizon model: PASS.** At the continuously propagated 0.7-s end-to-end horizon, the frozen PR predictor plus the 12-node DDER gives 11.491-mm UAV position RMSE, 4.112° UAV orientation RMSE, 53.286-mm distributed-marker RMSE, and 103.716-mm c10 error. Both cable-eligible validation episodes remain finite, with no coupled-rollout failure or reset. These values pass the predeclared usability thresholds; the excluded fig8_003 episode remains explicitly initialization-ineligible under the unchanged 10-mm rule.

The spatial reduction did **not** yield a large measured grid-throughput improvement in this full-episode fitting path. A 12-node grid pass averaged 86.41 s versus 90.97 s for the saved 21-node fit, a 5.02% reduction. Total fitting time fell 36.68% (172.82 s versus 272.91 s), principally because the frozen adaptive rule stopped after two passes instead of the old fit's three. No standalone MPPI benchmark was run, so this milestone does not claim a measured planner speedup.

Long-horizon drift remains reported in the raw validation artifact, but it is not the primary acceptance criterion for the approximately 0.7-s planning horizon. The protected `fig8vertical_002` take remains sealed.
