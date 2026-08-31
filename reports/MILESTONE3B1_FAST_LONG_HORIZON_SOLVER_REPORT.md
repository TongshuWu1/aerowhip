# Milestone 3B.1 — Structure-Exploiting Long-Horizon Solver Report

**Repository:** `C:\Users\wts28\Documents\PHD\particle_filter_cable_project`  
**Date:** 2026-08-28  
**Model:** `aerial_cable_attitude_coupled_v1`  
**Identification methodology:** `full_episode_constrained_multiple_shooting_v1`  
**Solver candidate:** `sparse_ggn_sqp_v1`  
**Final status:** **STOPPED AT THE LINEARIZATION RUNTIME GATE**

## Executive outcome

The Milestone 3B.1 profiling and A-versus-B linearization microbenchmark were completed on the RTX 4080. They show that the earlier failure was not caused by shot-by-shot Python propagation: the existing Milestone 3B evaluator already groups equal-length shooting segments into one batched production rollout.

The bottleneck is exact differentiation through each 100-step production DDER shot:

| Primitive, three-second synthetic problem | Measured time |
|---|---:|
| Batched three-segment forward graph | 33.328 s |
| Observation-loss construction | 0.0046 s |
| Continuity-defect construction | 0.0015 s |
| Whole-problem backward | 60.983 s |
| Old forward + backward primitive | 94.352 s |
| One exact parameter JVP, reverse-over-reverse | 193.098 s |
| One exact parameter JVP, composable forward AD | 137.191 s |
| Vectorized `jacfwd`, all seven global directions | 162.427 s |

The final `jacfwd` measurement includes only seven global parameter directions. It does not include the 252 shooting-state directions, observation-residual Jacobian blocks, sparse KKT assembly, a sparse solve, nonlinear trial evaluations, or multiple nonlinear iterations.

Therefore neither allowed implementation is plausibly capable of completing the unchanged synthetic recovery problem within the hard ten-minute budget:

- explicit local Jacobians require substantially more derivative directions than the measured seven-direction lower-bound workload;
- matrix-free GGN/SQP requires repeated JVP and VJP products, and one JVP alone costs 137.191 s.

The milestone's wall-clock stop rule was applied before building or launching an optimizer known to exceed the budget.

- Synthetic recovery: **NOT RUN after the failed linearization gate**.
- Real initialization and joint physical fit: **NOT RUN**.
- Provisional validation: **NOT RUN**.
- Residual refit: **NOT RUN**.
- `fig8vertical_002`: **NOT PREDICTIVELY EVALUATED**.
- MPPI: **NOT RUN**.
- New pretest freeze: **NOT CREATED**.

## A. Previous bottleneck

Milestone 3B used an augmented-Lagrangian objective with eight outer iterations and 100 Adam updates per outer iteration over approximately 22,987 shooting-state variables plus seven physical parameters. Its second three-second synthetic run remained finite but was terminated after 65 min 54 s without completing the scheduled 800 updates.

The new bounded profile measured one old optimization primitive directly:

```text
forward evaluation:          33.369 s
backward gradient:           60.983 s
forward + backward:          94.352 s
```

At 800 updates, that primitive alone implies roughly 21 hours before optimizer overhead. The earlier 66-minute stop was therefore not anomalous.

CUDA peak allocation during the bounded profile was approximately 496 MB and peak reserved memory was approximately 501 MB. The earlier Milestone 3B process reached approximately 17 GB private process memory; `psutil` was unavailable in the bounded rerun, so that value is retained only as a prior observation, not reported as a new measurement.

## B. Runtime profiling

### Detailed forward breakdown

| Component | Time [s] | Share of measured forward |
|---|---:|---:|
| Shooting-state physical retraction | 0.0301 | 0.09% |
| Batched segment propagation | 33.3284 | 99.88% |
| Observation loss | 0.0046 | 0.014% |
| Observation aggregation | 0.0002 | <0.001% |
| Continuity defects | 0.0015 | 0.004% |
| Unattributed/synchronization | 0.0043 | 0.013% |
| Total | 33.3691 | 100% |

This eliminates observation masks, hierarchical aggregation, continuity concatenation, and Python shot dispatch as meaningful primary bottlenecks.

### Existing batching audit

`MultipleShootingProblem.evaluate` already:

1. groups every `SegmentRecord` by `step_count`;
2. stacks all initial UAV and cable states for a group;
3. stacks commands as `[time, segment, command_dimension]`;
4. executes one batched production `CoupledSimulator.rollout` for the group.

For the synthetic problem, all three 100-step shots are propagated together. Reimplementing the requested batching would duplicate existing behavior and would not address the measured 33.3-s differentiable rollout.

## C. A-versus-B solver decision

### Option A — explicit local Jacobians plus sparse SQP/KKT

The smallest synthetic problem has:

```text
7 global parameter directions
2 internal shooting boundaries × 126 tangent coordinates
= 259 nonlinear directions
```

Vectorized `torch.func.jacfwd` required 162.427 s for only the seven global directions and a 253-value diagnostic output. A production explicit factorization would additionally require local observation residual Jacobians and local dynamics/continuity blocks for the 252 state directions.

Even without assuming linear scaling with direction count, 162 s for this strict subset is not compatible with several nonlinear SQP iterations inside ten minutes. The environment also does not currently contain SciPy, so a `scipy.sparse` KKT solve is not available without adding another dependency. This dependency issue is secondary: derivative construction fails the runtime gate before sparse factorization matters.

**Decision:** Option A rejected by measured derivative cost.

### Option B — matrix-free JVP/VJP SQP/KKT

The original reverse-over-reverse exact JVP took 193.098 s. A solver-only, `torch.func`-composable bending-force path was then added. It computes the gradient of the identical production bending/twist energy, disables the pointer-based fused projection only while transformed dual tensors are active, and leaves the normal production path unchanged.

With that path:

```text
one exact forward-mode JVP = 137.191 s
```

A matrix-free Gauss-Newton/KKT product requires at least one JVP plus a VJP. A useful Krylov solve requires repeated products. Even three JVPs alone consume 411.6 s; adding VJPs, trial rollouts, globalization, and another nonlinear iteration exceeds the ten-minute complete-recovery budget.

**Decision:** Option B rejected by measured matrix-vector-product cost.

### Numerical equivalence of the transform-compatible force path

The transform-compatible path is not a new force law. Both implementations return:

```text
-d E_bending/twist(q) / d q
```

from the same curvature, stiffness, dual-length, terminal, and twist expressions. It is disabled by default and was introduced only to make the microbenchmark scientifically meaningful. A CUDA regression test compares its rollout against the existing production autograd path.

No sparse GGN/SQP optimizer was promoted or exposed as a fitting backend.

## D. Synthetic recovery

The unchanged truth and initialization remain:

| Parameter | Truth | Initialization |
|---|---:|---:|
| `K_p` | 4.0 | 3.0 |
| `K_v` | 5.0 | 6.0 |
| `k_a` | 1.10 | 0.88 |
| `K_R` | 60.0 | 45.0 |
| `K_omega` | 12.0 | 15.0 |
| `EI` | 2.4e-6 | 3.6e-6 |
| `Cb` | 4.2e-8 | 2.52e-8 |

The problem was not shortened or simplified. However, a recovery optimization was not launched after the derivative microbenchmark demonstrated that even one incomplete linearization consumed a substantial fraction of the entire ten-minute budget.

| Gate | Result |
|---|---|
| Same scientific synthetic problem preserved | YES |
| Finite JVP/Jacobian primitives | YES |
| Complete sparse GGN/SQP iteration practical | NO |
| Complete recovery within 10 min plausible | NO |
| Synthetic recovery run | NOT RUN |
| Seven-parameter recovery | NOT ESTABLISHED |
| Continuity convergence | NOT ESTABLISHED |

This is a runtime-gate failure, not a parameter-recovery result.

## E. Real initialization

**NOT RUN.**

The UAV-only long-horizon initialization and the cable EI/Cb initialization were correctly gated behind successful synthetic recovery. No new initialization values exist.

## F. Final joint seven-parameter fit

**NOT RUN.**

No real-data value, bound hit, objective, continuity metric, gradient, or runtime is reported for `K_p`, `K_v`, `k_a`, `K_R`, `K_omega`, `EI`, or `Cb`.

The existing Milestone 3A values remain historical short-window results and were not silently promoted to the new attitude-coupled model.

## G. Parameter conditioning

**NOT RUN.** No converged Gauss-Newton system exists from which to compute a seven-parameter conditioning diagnostic.

## H. Physical-only long-horizon validation

**NOT RUN.** `fig8_003` and `osc_003` were not predictively evaluated because the physical fit did not pass its prerequisite gate.

## I. Residual refit

**NOT RUN.** The causal 100-ms, `90 -> 32 -> 32 -> 3` translational residual remains unchanged and locked. It was not renormalized, retrained, or attached to an unvalidated physical model.

## J. Physics-plus-residual validation

**NOT RUN.** No Model P versus Model PR long-horizon comparison exists.

## K. Runtime conclusion

The requested solver change alone is insufficient because the exact local linearization backend is itself too slow. Sparse assembly and a better KKT solver cannot remove a 137–162 s derivative primitive.

The next engineering decision must target the differentiable DDER linearization implementation while preserving the model—for example, an analytically or custom-autograd-defined local DDER tangent/adjoint operator, verified against the production step. That is a separate implementation milestone and must be approved explicitly because it is more invasive than replacing Adam with sparse SQP.

No claim is made that finite differences, shorter shots, relaxed continuity, reduced DDER fidelity, or measured resets would be acceptable substitutes. None were used.

## L. Protected test

`fig8vertical_002` **WAS NOT PREDICTIVELY EVALUATED**.

It remains excluded from fitting, normalization, solver selection, runtime decisions, validation metrics, and model selection. No pre-protected-test freeze was created.

## Artifacts

Artifacts are stored in:

```text
data/fit_results_long_horizon/milestone3b1_sparse_ggn_sqp_v1/
```

Measured artifacts:

- `solver_profile_before.json`
- `linearization_microbenchmark.json`
- `forward_ad_microbenchmark.json`
- `jacfwd_microbenchmark.json`
- `runtime_summary.json`

Explicit stopped/not-run artifacts are also present for synthetic recovery, physical initialization, joint fitting, conditioning, validation, residual fitting, and physics-plus-residual validation. These placeholders prevent absence from being mistaken for a completed result.

## Verification

| Check | Result |
|---|---|
| Existing eight structural tests | PASS |
| New transform-compatible force equivalence check | PASS |
| Focused long-horizon/data/GUI suite | 21/21 PASS |
| Full repository suite | 77/77 PASS in 51.55 s |
| Python compilation | PASS |
| `git diff --check` | PASS; existing line-ending notices only |
| Protected-test action | Disabled |
| Real fit/residual/validation/MPPI | NOT RUN |

## Final stop

Milestone 3B.1 stopped at its required wall-clock gate. The scientific model, PhysicalEpisode semantics, shooting interval, DDER fidelity, observation masks, continuity tolerances, target dataset roles, and protected-test boundary were preserved. No real fitting or downstream scientific experiment was performed.
