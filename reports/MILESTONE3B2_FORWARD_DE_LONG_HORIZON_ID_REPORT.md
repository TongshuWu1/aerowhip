# Milestone 3B.2 — Forward-Only Long-Horizon Identification Report

**Repository:** `C:\Users\wts28\Documents\PHD\particle_filter_cable_project`  
**Methodology candidate:** `full_episode_forward_de_v1`  
**Final status:** **STOPPED AT THE FORWARD-THROUGHPUT GATE**

## Executive outcome

The production-equivalent no-gradient population evaluator was implemented and verified. Candidate batching is highly effective: P=1 through P=64 take nearly the same wall time. However, one complete P=32 Training generation required **159.651 s**, exceeding the mandatory 60-second stop threshold.

Consequently:

- Differential Evolution implementation/run: **NOT STARTED**.
- Synthetic DE recovery: **NOT RUN**.
- Real staged or seven-parameter fit: **NOT RUN**.
- Provisional Validation: **NOT PREDICTIVELY EVALUATED**.
- Residual refit: **NOT RUN**.
- New freeze: **NOT CREATED**.
- `fig8vertical_002`: **NOT PREDICTIVELY EVALUATED**.

## A. Why the method was evaluated

The differentiable multiple-shooting backend remains preserved as an experimental historical backend blocked by DDER derivative cost. This milestone tested whether full-episode single shooting, a seven-dimensional parameter population, and the accepted forward production DDER could avoid that graph cost.

The scientific model and episode semantics were unchanged.

## B. Forward backend

```json
{
  "damping_backend": "pcg32_experimental",
  "device": "cuda",
  "dtype": "torch.float32",
  "node_count": 21,
  "pinned_start_nodes": 2,
  "position_projection_iterations": 4,
  "production_fused_forward_active": true,
  "projection_backend": "fixed_projection_supported_nodes",
  "substeps": 3
}
```

The evaluator calls the existing `CoupledSimulator` production transition under `torch.no_grad()`. It stores only running observation sums and current state. Commands, observations, masks, times, marker indices, and initialization states are cached on device.

Population parameters are one vector per simulator batch row. Scalar production parameters remain supported and regression-equivalent.

## C. Equivalence verification

- Slim streaming loss equals the full saved production rollout within float32 tolerance.
- A batched population equals independent candidate evaluation within float32 tolerance.
- Scalar and one-row batched parameters produce exactly equal saved trajectories.
- A deliberately runaway candidate is assigned the deterministic finite penalty without changing the valid candidate.
- Cached arrays exactly equal the declared float32 conversion of processed source arrays.

## D. Precision audit

```json
{
  "cable_position_discrepancy_m": {
    "final_maximum_absolute": 3.399675553850123e-05,
    "maximum_absolute": 3.399675553850123e-05,
    "rmse": 4.091699119536423e-06
  },
  "cable_velocity_discrepancy_mps": {
    "final_maximum_absolute": 0.00032925432944108657,
    "maximum_absolute": 0.0003693520138350337,
    "rmse": 5.381606358921901e-05
  },
  "objective": {
    "torch.float32": 5.217914917921007e-07,
    "torch.float64": 1.1631314387713492e-08
  },
  "objective_absolute_difference": 5.101601774043873e-07,
  "runtime_s": {
    "torch.float32": 2.381376600009389,
    "torch.float64": 38.62119919998804
  },
  "torch.float32": {
    "damping_backend": "pcg32_experimental",
    "device": "cuda",
    "dtype": "torch.float32",
    "node_count": 21,
    "pinned_start_nodes": 2,
    "position_projection_iterations": 4,
    "production_fused_forward_active": true,
    "projection_backend": "fixed_projection_supported_nodes",
    "substeps": 3
  },
  "torch.float64": {
    "damping_backend": "reference_pytorch_direct",
    "device": "cuda",
    "dtype": "torch.float64",
    "node_count": 21,
    "pinned_start_nodes": 2,
    "position_projection_iterations": 4,
    "production_fused_forward_active": false,
    "projection_backend": "reference_pytorch_projection",
    "substeps": 3
  },
  "uav_orientation_component_discrepancy": {
    "final_maximum_absolute": 7.719785186099948e-08,
    "maximum_absolute": 2.497761724118419e-07,
    "rmse": 6.188003350502334e-08
  },
  "uav_position_discrepancy_m": {
    "final_maximum_absolute": 2.483025887123347e-07,
    "maximum_absolute": 6.364043019235766e-07,
    "rmse": 2.254939838881755e-07
  },
  "uav_velocity_discrepancy_mps": {
    "final_maximum_absolute": 7.34601639296173e-07,
    "maximum_absolute": 1.2735761381055255e-06,
    "rmse": 3.626279274079705e-07
  }
}
```

Float64 uses the generic reference PyTorch path, not fused production damping/projection, and is far slower. It was retained only as a numerical audit; the forward feasibility benchmark used the planning-compatible float32 production backend.

## E. Forward benchmark

### Three-second synthetic episode

| P | wall [s] | physical s/wall s | candidates/s | candidate-physical s/wall s | peak allocated [MiB] | invalid |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.528 | 1.171 | 0.396 | 1.2 | 12.4 | 1 |
| 8 | 2.479 | 1.194 | 3.227 | 9.6 | 12.5 | 7 |
| 32 | 2.478 | 1.194 | 12.912 | 38.2 | 12.6 | 28 |
| 64 | 2.501 | 1.183 | 25.588 | 75.7 | 12.8 | 57 |

### Representative real episode

Representative episode: `osc_001__episode_000`, 14.61 physical seconds.

| P | wall [s] | physical s/wall s | candidates/s | candidate-physical s/wall s | peak allocated [MiB] | invalid |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.093 | 1.208 | 0.083 | 1.2 | 12.4 | 1 |
| 8 | 12.236 | 1.194 | 0.654 | 9.6 | 12.5 | 7 |
| 32 | 12.302 | 1.188 | 2.601 | 38.0 | 12.6 | 30 |
| 64 | 12.690 | 1.151 | 5.043 | 73.7 | 12.8 | 62 |

### Complete Training objective

| P | Training physical seconds | wall [s] | candidates/s | candidate-physical s/wall s | peak allocated [MiB] | invalid |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 188.04 | 159.651 | 0.200 | 37.7 | 12.6 | 31 |

The deterministic broad-bound populations contain many unstable candidates, as expected for unrestricted long-horizon single shooting. The historical five-parameter UAV values were fitted under the earlier direct-effective translation model and are not stable under the newer attitude/thrust-coupled full-episode model; on `osc_001__episode_000` they first exceeded the deliberately loose 1000 m/s runaway gate at 1.43 s. This is not treated as a physical fit result.

To check whether invalid-candidate replacement made the benchmark artificially slow or fast, one continuously valid Sobol candidate was rerun alone on the same 14.61-s episode. It required **11.782 s**, versus 12.093 s for the invalid historical-baseline row. Therefore the approximately 1.2 simulated-second/wall-second throughput is representative of valid forward propagation as well.

Training episodes cached and propagated:

- `fig8_001__episode_000`: 40.49 s (training)
- `fig8_002__episode_000`: 7.20 s (training)
- `fig8_002__episode_001`: 16.24 s (training)
- `fig8_002__episode_002`: 4.40 s (training)
- `fig8vertical_001__episode_000`: 11.25 s (training)
- `fig8vertical_001__episode_001`: 1.21 s (training)
- `fig8vertical_001__episode_002`: 5.93 s (training)
- `fig8vertical_001__episode_003`: 3.18 s (training)
- `fig8vertical_001__episode_004`: 18.92 s (training)
- `fig8vertical_001__episode_005`: 0.01 s (training)
- `fig8vertical_001__episode_006`: 0.01 s (training)
- `fig8vertical_001__episode_007`: 0.01 s (training)
- `fig8vertical_001__episode_008`: 9.57 s (training)
- `fig8vertical_001__episode_009`: 17.35 s (training)
- `osc_001__episode_000`: 14.61 s (training)
- `osc_001__episode_001`: 3.25 s (training)
- `osc_002__episode_000`: 5.44 s (training)
- `osc_002__episode_001`: 0.04 s (training)
- `osc_002__episode_002`: 0.01 s (training)
- `osc_002__episode_003`: 1.67 s (training)
- `osc_002__episode_004`: 27.25 s (training)

## F. DE feasibility

```json
{
  "acceptable_gate_s": 30.0,
  "cached_parent_fitness": true,
  "de_implemented": false,
  "de_run": false,
  "gate_passed": false,
  "hard_stop_gate_s": 60.0,
  "measured_generation_s_P32": 159.6507870000205,
  "preferred_gate_s": 15.0,
  "projected": {
    "100": {
      "minutes": 266.0846450000342,
      "seconds": 15965.07870000205
    },
    "25": {
      "minutes": 66.52116125000855,
      "seconds": 3991.2696750005125
    },
    "50": {
      "minutes": 133.0423225000171,
      "seconds": 7982.539350001025
    }
  },
  "schema": "milestone3b2_de_runtime_estimate_v1",
  "trial_population_evaluations_per_generation": 1
}
```

The measured full generation fails the predeclared feasibility gate. Cached-parent DE would still require one complete trial-population evaluation per generation. Even 25 generations would exceed the initial 60-minute engineering limit.

No optimizer was implemented or launched after this result.

## G. Scientific status

The result does **not** show that derivative-free physical identification is scientifically inadequate. It shows that the current production forward DDER remains too slow for repeated full-Training population generations under the stated runtime budget.

The key remaining cost is the sequential 100-Hz forward transition: three DDER substeps per frame, bending-force autograd inside each substep, fused damping, and fused four-position-plus-one-velocity projection. Population arithmetic scales well, but there are still 18,804 sequential episode frames per generation.

## H. Required stop

Milestone 3B.2 stopped before DE, validation, residual refitting, or freezing. No fidelity, episode duration, observation boundary, loss, parameter bound, or dataset role was altered to make the benchmark pass.

`fig8vertical_002` **WAS NOT PREDICTIVELY EVALUATED**.

## I. Verification

| Check | Result |
|---|---|
| Forward evaluator/full production trajectory and loss | PASS |
| Candidate batch/independent candidates | PASS |
| Scalar/one-row batched parameters, all reported state fields | PASS, exact |
| Cached tensors/source data after declared dtype conversion | PASS, exact |
| Invalid candidate isolation | PASS |
| Production fused backend assertion | PASS |
| Python compilation | PASS |
| Full repository suite | **83/83 PASS in 49.16 s** |
| Protected test predictive action | NOT PERFORMED |

## Artifacts

- `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results_forward_de\2026-08-28T182946Z_forward_gate\forward_backend_audit.json`
- `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results_forward_de\2026-08-28T182946Z_forward_gate\forward_benchmark.json`
- `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results_forward_de\2026-08-28T182946Z_forward_gate\precision_audit.json`
- `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results_forward_de\2026-08-28T182946Z_forward_gate\de_runtime_estimate.json`
- `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results_forward_de\2026-08-28T182946Z_forward_gate\dataset_snapshot.json`
- `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results_forward_de\2026-08-28T182946Z_forward_gate\environment.json`
- `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results_forward_de\2026-08-28T182946Z_forward_gate\verification.json`
- `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\fit_results_forward_de\2026-08-28T182946Z_forward_gate\valid_candidate_runtime_diagnostic.json`
