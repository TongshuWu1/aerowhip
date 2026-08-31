# Milestone 3C — Decomposed Refit and Long-Horizon Validation

Repository: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project`  
Methodology: `decomposed_full_episode_id_v1`  
Model: `aerial_cable_attitude_coupled_v1`  
Artifact: `data/fit_results_decomposed/2026-08-28T200529.563705+0000_e828f6b1`  
Status: **FITTING COMPLETE; PROVISIONAL VALIDATION COMPLETED WITH EXPLICIT EPISODE FAILURES; PRETEST FREEZE CREATED**

## A. Executive result

The prescribed decomposition was executed:

1. fit the attitude-coupled UAV model on complete Training `PhysicalEpisode`s;
2. fit `EI` and `Cb` with the measured Motive UAV boundary;
3. validate DDER conditionally on that measured boundary;
4. train a new causal UAV residual from zero on the suffixes permitted by `PhysicalEpisode.residual_eligible_start_index`;
5. evaluate physics-only and physics-plus-residual UAV models;
6. evaluate command-to-UAV-to-DDER prediction without a measured boundary after initialization;
7. freeze the result before the protected test.

All fits completed inside the 90-minute execution policy. Validation was attempted on every eligible provisional-validation episode. It was **not numerically complete**: the physics-only UAV model became non-finite on the long `fig8_003` episode, and the short second `fig8_003` episode exceeded the frozen 10 mm cable-initialization RMSE gate. These failures are retained as results; no episode was shortened, reset, or replaced.

The provisional classification is:

> **BOTH REMAIN LIMITING**

Evidence:

- the full-episode UAV physics model is not stable/general across all validation episodes;
- conditional measured-boundary DDER validation is already inaccurate (222.2 mm distributed-marker and 378.8 mm tip RMSE);
- the causal residual materially improves the paired UAV and cable prediction that can be compared, and it prevents the long `fig8_003` UAV divergence, but the resulting cable error remains large;
- `EI` reaches the upper boundary of the frozen search range, so its fitted value is boundary-limited rather than an interior optimum.

No additional physics or model architecture was introduced.

## B. Data and authoritative episode units

| Take | Role | PhysicalEpisodes | Physical duration [s] |
|---|---|---:|---:|
| `fig8_001` | Training | 1 | 40.49 |
| `fig8_002` | Training | 3 | 27.84 |
| `fig8vertical_001` | Training | 10 | 67.44 |
| `osc_001` | Training | 2 | 17.86 |
| `osc_002` | Training | 5 | 34.41 |
| **Training total** |  | **21** | **188.04** |
| `fig8_003` | Provisional Validation | 2 | 56.41 |
| `osc_003` | Provisional Validation | 1 | 15.50 |
| **Validation total** |  | **3** | **71.91** |
| `fig8vertical_002` | Protected Test | not evaluated | not used |

Each physical episode was initialized once and propagated continuously to its original end. Overlapping one-second historical prediction windows were not used as fitting units.

Top-level training losses are take-balanced: observations are aggregated within a take and then physical takes are averaged equally.

## C. Causal residual coverage amendment

The physical episode boundaries were not changed for residual history. Residual eligibility came only from the existing production field:

`PhysicalEpisode.residual_eligible_start_index`

| Split | PhysicalEpisodes | Residual-eligible suffixes | Physical duration [s] | Eligible duration [s] | Excluded [s] |
|---|---:|---:|---:|---:|---:|
| Training | 21 | 16 | 188.04 | 186.36 | 1.68 |
| Validation | 3 | 3 | 71.91 | 71.61 | 0.30 |

Five microscopic Training episodes have no eligible residual suffix under the general rule:

- `fig8vertical_001` episodes 005, 006, and 007;
- `osc_002` episodes 001 and 002.

These IDs are not special-cased in code. No zero padding, repeated state, fabricated command, or command-gap crossing was used. After suffix initialization, the residual FIFO uses recorded commands and simulated position/velocity only. Physics-only full-episode metrics remain separate from residual-suffix metrics.

## D. UAV physics refit

### Model

The fitted UAV model is the existing attitude-coupled effective model:

```text
e_p = p_cmd - p
e_v = v_cmd - v
a_ctrl = K_p e_p + K_v e_v + k_a a_cmd
f_des = a_ctrl + g e_z
b3_des = normalize(f_des)
R_des = attitude_from(b3_des, yaw_cmd)
omega_dot = K_R e_R(R_des, R) - K_omega omega
a_phys = ||f_des|| (R e_z_body) - g e_z
p_dot = v
v_dot = a_phys
```

The same simulated attitude determines realized acceleration, attachment offset, and cable root tangent.

### Optimization

- Method: `full_episode_sobol_adam_v1`.
- Bounds were unchanged.
- Seed: 42.
- Requested initial Sobol block: 64.
- Only one of the first 64 candidates was finite over all 188.04 s of Training data, so the same deterministic Sobol stream was extended once to 128 candidates to obtain the required two finite refinement starts.
- Two Adam refinements were run.
- Each refinement used 60 updates rather than the configured ceiling of 300 because of the milestone-wide 90-minute policy.
- Learning rate: 0.01; gradient clipping: 10.
- Selection used Training objective only.
- Runtime: 1290.38 s.
- Best take-balanced Training objective: 1.997603.

### Parameters

The historical values below came from the earlier short-window model and are context only.

| Parameter | Historical short-window | Bound | Milestone 3C fitted | Bound status |
|---|---:|---:|---:|---|
| `K_p` | 24.630215 | [1, 60] | 4.02009778 | near lower side; not at bound |
| `K_v` | 24.363403 | [0.5, 30] | 12.05728865 | interior |
| `k_a` | 1.051226 | [0.1, 2] | 0.73273015 | interior |
| `K_R` | 59.591014 | [1, 80] | 69.18419376 | interior |
| `K_omega` | 8.910099 | [0.5, 30] | 11.45658317 | interior |

### Full-episode Training metrics

| Take | Position RMSE [mm] | Orientation RMSE [deg] |
|---|---:|---:|
| `fig8_001` | 42.624 | 5.384 |
| `fig8_002` | 57.216 | 6.281 |
| `fig8vertical_001` | 43.491 | 4.555 |
| `osc_001` | 111.163 | 7.434 |
| `osc_002` | 50.849 | 3.750 |
| **Equal-take aggregate** | **66.219** | **5.630** |

### Physics-only full-episode provisional validation

This validation is **incomplete**.

| Take / episode | Outcome | Position RMSE [mm] | Orientation RMSE [deg] | Terminal position [mm] |
|---|---|---:|---:|---:|
| `fig8_003` episode 000, 55.33 s | non-finite physics-only rollout | n/a | n/a | n/a |
| `fig8_003` episode 001, 1.08 s | completed | 18.332 | 6.820 | 22.217 |
| `osc_003` episode 000, 15.50 s | completed but diverged severely late | 4756.056 | 12.655 | 84849.131 |
| Reported equal-take aggregate over surviving data | partial only | 3363.065 | 10.165 | 59997.398 |

The aggregate above must not be read as a complete validation score. It combines the surviving short `fig8_003` episode with `osc_003`; the long `fig8_003` failure remains a separate failure outcome.

| Lead [s] | Position RMSE [mm] | Orientation RMSE [deg] | Takes contributing |
|---:|---:|---:|---:|
| 0.10 | 7.605 | 0.805 | 2 |
| 0.25 | 23.983 | 6.063 | 2 |
| 0.50 | 34.792 | 6.917 | 2 |
| 1.00 | 32.309 | 6.475 | 2 |
| 2.00 | 140.373 | 6.802 | 1 |
| 4.00 | 87.976 | 11.386 | 1 |
| 8.00 | 178.287 | 4.035 | 1 |

The early lead-time behavior can be reasonable even though continuous long-horizon propagation later diverges. This is precisely why the physical-episode evaluation differs from the old window result.

## E. Cable physics fit with measured UAV boundary

### Boundary and loss

For this subsystem fit only, the measured Motive UAV pose drives the production rigid attachment:

```text
r0 = p_uav + R_uav d_attachment
r1 = r0 + l0 normalize(R_uav t_attachment)
```

The measured boundary is interpolated through the production DDER substep path. Cable markers `c1 ... c10` are compared with DDER nodes `2, 4, ..., 20` using the existing validity masks and pseudo-Huber loss. The cable is initialized once per complete eligible episode and is never reset from measurements afterward.

### Search and result

- Method: `measured_boundary_log_grid_v1`.
- Three deterministic 8 × 8 log-space passes.
- `EI` bound: `[1e-8, 4e-4]`.
- `Cb` bound: `[1e-10, 1.5e-5]`.
- Runtime: 272.91 s.

| Pass | Best EI | Best Cb | Objective | Runtime [s] |
|---:|---:|---:|---:|---:|
| 1 | 4.000000e-4 | 4.979756e-7 | 1.30963264e-4 | 92.40 |
| 2 | 4.000000e-4 | 6.351007e-7 | 1.30946719e-4 | 90.16 |
| 3 | 4.000000e-4 | 7.823236e-7 | 1.30944929e-4 | 90.33 |

Final values:

```text
EI = 4.000000000e-4 N m²
Cb = 7.823235937e-7 N m² s
```

`EI` is exactly at the upper search boundary. The final objective changes only slightly across refinement. The present data/search therefore do **not** establish a well-contained interior EI optimum. The full grids and `cable_loss_landscape.png` are saved.

During evaluation, `fig8_003` episode 001 had cable initialization RMSE 11.709 mm, above the frozen 10 mm gate. The entire episode was recorded as cable-ineligible; it was not trimmed.

## F. Conditional cable validation

These numbers answer: “How well does DDER predict the cable when the true measured UAV boundary is supplied?” They are not command-to-cable predictions.

| Take | Distributed marker RMSE [mm] | Tip RMSE [mm] | Terminal tip error [mm] |
|---|---:|---:|---:|
| `fig8_003` | 272.562 | 455.214 | 47.365 |
| `osc_003` | 156.372 | 282.367 | 161.958 |
| **Equal-take aggregate** | **222.196** | **378.782** | **119.319** |

Lead-time tip error is 5.28, 12.63, 54.36, 42.63, 175.03, 248.25, and 365.43 mm at 0.10, 0.25, 0.50, 1, 2, 4, and 8 s respectively.

Because this error is present even with the real UAV boundary, cable physics/initialization/observation representation remains a material limitation. Upstream UAV prediction cannot explain it.

## G. Causal UAV residual refit

### Frozen architecture

- Method: `full_episode_causal_residual_v1`.
- Inputs per sample: simulated `e_p`, simulated `e_v`, and recorded `a_cmd`.
- History: 10 causal samples (100 ms at 100 Hz), initialized by the production eligibility rule.
- Network: `90 → 32 → SiLU → 32 → SiLU → 3`.
- Output: translational acceleration residual `Delta_a` in m/s².
- The residual is added only to realized translational acceleration; it does not alter `a_ctrl`, `R_des`, or attitude dynamics.
- Nominal UAV parameters were frozen.
- Historical residual weights were not reused.
- Output layer was initialized to zero.
- Normalization was recomputed from the new nominal model on Training suffixes only.
- Magnitude regularization: 0.01; smoothness regularization: 0.05.
- Adam learning rate: 0.001; gradient clipping: 5; seed: 42.
- 150 updates were executed rather than the configured ceiling of 500 under the total runtime policy.
- Runtime: 1847.22 s.
- Best full-Training objective: 0.417932 (initial: 1.9327).
- Parameter count: 4067.

### Same-suffix Training result

| Model | Position RMSE [mm] | Orientation RMSE [deg] |
|---|---:|---:|
| U — fitted physics | 64.569 | 5.878 |
| UR — fitted physics + residual | 20.401 | 5.624 |

### Fair paired provisional validation

Physics-only diverged on the 55.23 s `fig8_003` residual suffix, while UR completed it. Consequently, the raw U and UR validation aggregates cover different successful episode sets and are **not** a fair pair.

The exact common successful episode intersection is:

- `fig8_003__episode_001__residual_suffix` (0.98 s);
- `osc_003__episode_000__residual_suffix` (15.40 s).

| Model | Paired position RMSE [mm] | Paired orientation RMSE [deg] |
|---|---:|---:|
| U | 92.973 | 9.882 |
| UR | 41.997 | 8.092 |

On this limited common set, UR improves position RMSE by 54.8% and orientation RMSE by 18.1%.

| Lead [s] | U position [mm] | UR position [mm] | U orientation [deg] | UR orientation [deg] | Paired takes |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 6.098 | 4.721 | 2.410 | 2.579 | 2 |
| 0.25 | 20.704 | 18.331 | 6.162 | 3.474 | 2 |
| 0.50 | 27.607 | 29.075 | 5.766 | 4.410 | 2 |
| 1.00 | 93.119 | 14.800 | 4.235 | 16.560 | 1 |
| 2.00 | 157.935 | 37.850 | 4.161 | 3.662 | 1 |
| 4.00 | 64.288 | 46.911 | 11.991 | 8.668 | 1 |
| 8.00 | 147.438 | 36.788 | 6.066 | 3.498 | 1 |

The residual improves position at six of seven reported leads, but worsens the paired 0.5 s position metric and strongly worsens the 1 s orientation diagnostic. It should therefore be described as helpful but not uniformly better.

For completeness, UR completed all three residual-eligible UAV suffixes and its unequal-coverage validation aggregate was 53.386 mm and 10.137°. That number must not be compared directly with the U aggregate.

### Residual magnitude

| Split | RMS norm [m/s²] | X RMS | Y RMS | Z RMS | P95 norm | Maximum | Smoothness RMS delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 0.821 | 0.658 | 0.320 | 0.373 | 1.922 | 3.827 | 0.082 |
| Validation | 1.971 | 1.460 | 1.278 | 0.349 | 3.867 | 5.776 | 0.191 |

The residual remains below the predeclared 6 m/s² P95 and 12 m/s² maximum gates, and is temporally smooth relative to its RMS magnitude. Validation residual magnitude is substantially larger than Training, especially in X/Y, which is a distribution-shift warning.

## H. End-to-end command-to-UAV-to-DDER validation

No measured UAV boundary or measured UAV/cable state was used after initialization in this stage.

### Full-episode Model P

`fig8_003` episode 000 became non-finite; episode 001 failed the cable initialization gate. Only `osc_003` produced stored trajectory metrics, and that trajectory diverged severely late:

| Successful stored take | UAV position RMSE [mm] | UAV orientation [deg] | Cable markers [mm] | Tip [mm] | Terminal tip [m] |
|---|---:|---:|---:|---:|---:|
| `osc_003` | 4760.561 | 12.655 | 4553.188 | 4475.175 | 77.483 |

These are not complete two-take validation statistics.

### Fair same-suffix P versus PR

The only end-to-end suffix completed by both P and PR is `osc_003__episode_000__residual_suffix`. `fig8_003` episode 000 was completed only by PR, and episode 001 failed the same cable-initialization gate for both.

| Model on paired `osc_003` suffix | UAV position [mm] | UAV orientation [deg] | Cable markers [mm] | Tip [mm] |
|---|---:|---:|---:|---:|
| P | 129.909 | 12.428 | 319.773 | 515.204 |
| PR | 43.204 | 11.218 | 202.080 | 358.818 |

For this one paired physical episode, PR reduces UAV position error by 66.7%, distributed cable error by 36.8%, and tip error by 30.4%. This is promising but is not a broad validation claim.

PR also completed the long `fig8_003` suffix that P could not. Across the two PR-completed takes, the unequal-coverage aggregate is 53.568 mm UAV position, 10.170° orientation, 255.026 mm distributed cable, and 432.169 mm tip RMSE. This is reported as a stability/coverage result, not as a paired performance comparison.

The exact paired calculation is saved in `paired_same_suffix_comparison.json`.

## I. Error localization

The current evidence separates two failures:

1. **UAV/boundary model limitation.** Physics-only becomes non-finite on the long horizontal Figure-8 suffix and diverges badly late in `osc_003`. The residual stabilizes/completes the long Figure-8 suffix and improves the paired `osc_003` boundary prediction.
2. **Cable model/initialization limitation.** Even with the real measured UAV boundary, validation cable error is 222.2 mm distributed and 378.8 mm at the tip, so end-to-end cable error is not caused only by the UAV boundary.

The result does not identify which unmodeled mechanism is responsible. It does not justify adding a particular force, delay, cable reaction, drag term, or larger network without a separate scientific decision.

## J. Final provisional seven parameters

| Parameter | Value | Provenance |
|---|---:|---|
| `K_p` | 4.02009777865 | FullState command → measured UAV |
| `K_v` | 12.0572886500 | FullState command → measured UAV |
| `k_a` | 0.732730146833 | FullState command → measured UAV |
| `K_R` | 69.1841937593 | FullState command → measured UAV |
| `K_omega` | 11.4565831743 | FullState command → measured UAV |
| `EI` | 4.000000000e-4 | measured-boundary DDER; upper-bound limited |
| `Cb` | 7.823235937e-7 | measured-boundary DDER |

## K. Verification and implementation notes

Verified invariants:

- UAV fitting uses the same `FullStateUAVModel` executed by `CoupledSimulator`.
- Cable fitting uses the production rigid attachment and DDER implementation.
- Measured UAV boundary is used only for cable subsystem fitting/conditional validation.
- No measured cable reset occurs after initialization.
- End-to-end validation uses simulated UAV boundary only.
- Residual inputs use simulated UAV state after initialization.
- P and PR use the same DDER implementation and fitted EI/Cb.
- Residual eligibility is derived from `residual_eligible_start_index`, not a hard-coded offset.
- `fig8vertical_002` was not used for fitting, normalization, validation, plots, or model selection.

Two evaluator integrity corrections were required during execution:

1. missing cable-marker NaNs were masked before robust-distance evaluation so `NaN × 0` could not contaminate the EI/Cb population objective;
2. a whole episode that failed the frozen cable-initialization gate is now recorded as ineligible instead of aborting end-to-end validation.

Neither correction changes physics, model equations, loss weights, episode boundaries, parameter bounds, or optimizer settings.

## L. Runtime and reproducibility

Authoritative fitting-stage runtime:

| Stage | Runtime [s] |
|---|---:|
| UAV physics fit | 1290.38 |
| EI/Cb grid | 272.91 |
| UAV residual fit | 1847.22 |
| **Fit-stage total** | **3410.52 (56.84 min)** |

Validation/replay and the two corrected resume passes remained within the 5400 s milestone limit. Progress artifacts were written throughout.

Reproducibility:

- Git commit: `cbdb59b4f05472506599b6f7e08be08e05541ff3`.
- Working tree was dirty because of the intentional repository restructuring; its status hash is frozen.
- UAV/residual seed: 42.
- Residual network-state SHA-256: `f39b9e63c90f550bd07b6c9c2f91aec23469762aee9df6abc391334a5c75a18c`.
- Source aggregate SHA-256: `d34f1550d0d1e271ceb99298660d6028d18b5e23bcaead2fd0f917f96efc8d8f`.
- Freeze artifact-manifest SHA-256: `784fd0fe93281ce688ac5e4fd2f0d11ef6bcc759a4e46a137e8b3ab65d7e7a1a`.

## M. Freeze and protected test

Freeze:

`data/model_freezes/MODEL_FREEZE_DECOMPOSED_PRETEST`

The freeze includes the seven physical/effective parameters, residual weights, residual normalization, dataset-role snapshot, episode manifest, verification record, source hashes, and the exact residual-eligibility rule/source hash.

`fig8vertical_002` **WAS NOT PREDICTIVELY EVALUATED**.

Its future protocol is already frozen:

- physics-only may be reported on complete eligible physical episodes;
- physics versus physics-plus-residual must be compared on exactly the same `residual_eligible_start_index` suffixes;
- no convention may be changed after protected-test results are seen.

## N. Scientific decision

Do not promote the current model as sufficient for final prediction. Do not claim the residual solves the coupled-system problem. The correct provisional statement is:

> Long-horizon physical-episode evaluation reveals both UAV boundary-model instability and substantial measured-boundary cable-model error. The causal residual materially improves the limited paired validation and stabilizes one long validation motion, but coupled cable prediction remains inaccurate. Both subsystems remain limiting, and the protected test must stay sealed until the next model decision is reviewed.

Milestone 3C stops here. No MPPI, new physics, cable reaction, cable residual, architecture search, or protected-test evaluation was run.
