# Exact forward acceleration of the 11-node DDER–MPPI controller

Date: 26 August 2026  
Hardware: NVIDIA GeForce RTX 4080  
Software: PyTorch 2.11.0+cu128, CUDA runtime 12.8

## Executive conclusion

The production 11-node receding-horizon controller was accelerated without
changing the cable parameters, DDER equations, timestep, projection method,
fixed 60-iteration float64 PCG damping solve, MPPI cost, candidate count,
noise, temperature, contact logic, safety logic, or warm-start semantics.

The warmed MPC update improved from **589.90 ms mean / 590.21 ms median** to
**82.84 ms mean / 82.47 ms median**. This is a **7.12× mean speedup** and a
**7.16× median speedup**. The final p95 is **85.55 ms**.

The achieved planner-only computational rate is approximately **12.1 Hz**
using the mean time, or **11.7 Hz** using p95. The requested practical planning
target of 70–80 ms has therefore **not** been achieved: 10 Hz planning is now
computationally possible in isolation, but the current p95 leaves only about
14.4 ms of a 100 ms cycle for OptiTrack, state estimation, communication, and
jitter.

The strongest retained changes are:

1. a fixed-topology CUDA damping operator that constructs the unchanged
   objective curvature-rate Jacobian and the unchanged Kelvin–Voigt system,
   then performs the same 60 Jacobi-PCG iterations in float64;
2. a fused four-position-plus-one-velocity projection kernel implementing the
   existing chain Thomas method;
3. capture of the full fixed 35-step rollout rather than Python replay of a
   one-step graph;
4. one fused CUDA kernel for the unchanged MPPI event, contact, safety, and
   cost calculation.

The final implementation preserves the frozen candidate population's top-1,
top-5, and top-10 selections exactly. Valid-strike and contact-order
classification agreement are both 100%. The largest full-rollout discrepancy
among the frozen cases is 2.265 micrometres in node position and 29.64
micrometres/s in node velocity.

## A. Reference implementation

### A.1 Authoritative workload

The frozen reference configuration is:

| Setting | Value |
|---|---:|
| Cable topology | 11 DDER nodes, one attached root, free distal tip |
| Prediction horizon | 0.7 s |
| Physics and command rate | 50 Hz |
| Physics steps per rollout | 35 |
| Acceleration knots | 11 × 3 = 33 decision values |
| MPPI candidates | 2,048 |
| MPPI iterations | 2 |
| Sampled trajectories/update | 4,096 |
| DDER substeps | 1 |
| Position projections | 4 |
| Velocity projections | 1 |
| Damping solve | fixed 60-iteration Jacobi-PCG, float64 |
| Gradient guidance | disabled |
| Model mismatch | none: truth/model EI = 1, Cb = 1 |

No model parameter was refitted. The retained controller uses

* EI = 1.0154787051679222e-4 N m^2;
* Cb = 1.5e-5 N m^2 s.

### A.2 Reference timing

Twenty warmed reference updates produced:

| Statistic | Time |
|---|---:|
| Cold update | 1099.11 ms |
| Mean | 589.90 ms |
| Median | 590.21 ms |
| Standard deviation | 2.39 ms |
| Minimum | 584.73 ms |
| p95 | 591.86 ms |
| Maximum | 596.01 ms |
| Warm peak CUDA allocation | 69.88 MiB |

The frozen reference bundle contains seven representative one-step states,
nine complete 35-step rollouts, and one seeded 2,048-candidate MPPI
population. It records positions, velocities, damping solutions, constraint
residuals, controls, event metrics, costs, weights, and the weighted update.

### A.3 Frozen validation hierarchy

The validation tool checks:

1. one-step position, velocity, damping, attachment, and edge-length results;
2. complete rollout position/velocity and hard event metrics;
3. fixed-population cost and rank agreement;
4. importance weights and weighted knot update;
5. paired closed-loop outcomes under nominal and disturbed cable states.

## B. Winner-replay analysis

### B.1 Actual controller semantics

Within each MPPI iteration the importance-weighted nominal trajectory is the
center for the next proposal population. The controller separately retains the
lowest-cost sampled trajectory seen across all MPPI iterations. The returned
and executed `MppiPlan`, and therefore the shifted receding-horizon warm start,
use this global best sampled trajectory—not the final weighted nominal.

The selected controls were already propagated as part of a batch of 2,048.
However, production then propagates the same controls again at batch one and
uses that batch-one result for the reported prediction, hard diagnostics, and
cost.

### B.2 Attempted elimination and decision

Retaining the batched winner would have reduced the intermediate update to
about 440.35 ms, saving roughly 150 ms. It was not retained. The batch-2,048
and batch-one CUDA paths differed by approximately 5.36e-7 m in position and
5.72e-6 m/s in velocity. Although the control sequence was identical, deleting
the replay would change the production prediction and diagnostic semantics.

Result: **the replay is mathematically redundant for the selected controls but
not implementation-exact for the output trajectory. It remains enabled.**

## C. Fused exact damping implementation

### C.1 Preserved mechanics

The new damping operator evaluates the same objective curvature-rate Jacobian
and the same system

\[
A=M+\Delta t\,C_b J_\kappa^T WJ_\kappa.
\]

It preserves the existing tangent normalization, curvature-binormal
denominator clamp, corotational/objective derivative, boundary contribution,
mass diagonal, float32 system construction, float64 PCG recurrence, Jacobi
preconditioner, zero initial solution, fixed 60 iterations, and output cast to
float32.

The matrix is stored using five 3×3 block diagonals. This is not a numerical
approximation: the three-node curvature stencil makes all omitted entries of
`J_kappa^T W J_kappa` identically zero.

### C.2 GPU mapping

One CUDA block containing one warp handles one 30-DOF damping system/rollout.
The 30 free scalar DOFs map naturally to warp lanes. The exact Jacobian,
prescribed rate, five-block matrix representation, right-hand side, PCG
vectors, and Jacobi diagonal remain in shared memory or registers for the
fixed solve. Warp reductions implement the PCG dot products without separate
global reductions.

The operator is compiled at runtime with NVRTC and is compatible with CUDA
graph capture. The generic PyTorch implementation remains the fallback for
non-CUDA, non-float32, non-11-node, different-boundary, or non-60-iteration
cases.

### C.3 Timing

Two exact damping changes were measured independently:

| Stage | Warm mean | Stage speedup | Cumulative speedup |
|---|---:|---:|---:|
| Reference PyTorch damping | 589.90 ms | — | 1.00× |
| Fixed assembled-A/rhs 60-PCG kernel | 358.80 ms | 1.64× | 1.64× |
| Fused J_kappa + banded A/rhs + fixed PCG | 201.43 ms | 1.78× | 2.93× |

In the final controller, `fixed_damping_11node_60pcg` is called 105 times per
MPC update: 70 calls for the two sampled horizons and 35 for the retained
batch-one winner. It consumes 59.72 ms, which is **78.96% of summed CUDA kernel
time** and **72.10% of final wall-update time**.

### C.4 Numerical equivalence

Across the seven frozen one-step cases:

* maximum damping-solution difference: 8.94e-8 m/s;
* maximum final position difference: 1.19e-7 m;
* maximum final velocity difference: 2.44e-6 m/s;
* attached-node position and velocity differences remain at float32-level;
* edge-length residuals retain the same reference values to float32-level.

No EI or Cb adjustment was made.

## D. Fused projection implementation

### D.1 Preserved method

The new projection operator preserves, in the same sequence:

1. attachment clamping;
2. four position-level length projections;
3. the existing mass-weighted chain matrix;
4. the same regularization;
5. the same symmetric tridiagonal Thomas recurrence;
6. the same position corrections;
7. provisional velocity from the projected positions;
8. one velocity projection using the same Thomas method;
9. final attachment velocity clamping.

One block is assigned to one 11-node chain. Because this particular solve is
tiny and sequential, one active thread keeps all chain arrays local and fully
unrolls the four position iterations and velocity projection. This design was
measured faster than retaining the fragmented batched tensor operations.

### D.2 Timing

Adding fused projection reduced the warm mean from 201.43 ms to 94.50 ms:

* stage speedup: 2.13×;
* cumulative speedup: 6.24×.

After all changes, the projection kernel consumes 2.255 ms/update, or **2.98%
of summed CUDA kernel time** and **2.72% of final wall time**. Its Amdahl upper
bound is therefore only about 1.028× even if projection were eliminated.

## E. Reprofiled controller

### E.1 Stage-by-stage measurements

Every row is a separate process and contains one cold update followed by 20
warmed updates. The stage columns are cumulative.

| Stage | Cold ms | Mean ms | Median ms | Std ms | Min ms | p95 ms | Max ms | Peak MiB | Mean speedup vs prior | Cumulative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference | 1099.11 | 589.90 | 590.21 | 2.39 | 584.73 | 591.86 | 596.01 | 69.88 | — | 1.00× |
| Fixed PCG | 823.09 | 358.80 | 358.24 | 2.82 | 354.45 | 362.86 | 363.15 | 69.88 | 1.64× | 1.64× |
| Fused damping | 492.24 | 201.43 | 201.19 | 1.89 | 197.69 | 205.06 | 205.65 | 45.51 | 1.78× | 2.93× |
| Fused projection | 315.86 | 94.50 | 94.48 | 1.95 | 90.31 | 97.37 | 97.39 | 45.51 | 2.13× | 6.24× |
| Full 35-step graph | 585.05 | 88.42 | 88.81 | 1.61 | 85.72 | 90.43 | 91.72 | 40.54 | 1.07× | 6.67× |
| Fused event/cost | 551.95 | **82.84** | **82.47** | **1.83** | **80.34** | **85.55** | **87.23** | **25.90** | 1.07× | **7.12×** |

The initially low-priority event/cost path became material only after damping
and projection fusion: it measured about 7.1–8.8 ms in the preceding stages.
Fusing it at that point was therefore justified by the requested reprofile;
it did not add or remove any cost term.

### E.2 Final mutually exclusive GPU-kernel accounting

One warmed full update produced 75.63 ms of summed exclusive CUDA kernel time.

| Category | Calls | Exclusive device time | Kernel-time share | Whole-wall share |
|---|---:|---:|---:|---:|
| Fixed damping | 105 | 59.722 ms | 78.96% | 72.10% |
| Fixed projection | 105 | 2.255 ms | 2.98% | 2.72% |
| Fused hard event/cost | 3 | 0.424 ms | 0.56% | 0.51% |
| All other kernels | 10,110 | 13.233 ms | 17.50% | 15.97% |

The remaining `other` category includes unchanged bending-force/autograd
kernels, drone/attachment integration, state copies, control interpolation,
MPPI noise/weights, and framework operations. Independent one-step event
timings estimate the unchanged bending path at about 13 ms over the 105 physics
steps, but this estimate is not used as a mutually exclusive whole-update
percentage because the measurement graph introduces profiling overhead.

### E.3 Final update phases

Function-boundary CUDA events for one instrumented final update report:

| Phase | GPU time |
|---|---:|
| Two sampled DDER rollouts | 66.963 ms |
| Batch-one winner rollout | 10.421 ms |
| Three event/cost evaluations | 1.093 ms |
| Knot interpolation/clipping | 1.208 ms |
| Noise generation | 0.381 ms |
| Unclassified/overhead in event span | 3.497 ms |

The two sampled horizons execute 143,360 DDER step instances in 66.963 ms,
approximately **2.14 million DDER step instances/s**. Including the winner and
all other update work gives approximately 1.73 million step instances/s.

### E.4 Sample scaling

The optimized one-iteration diagnostic executes the actual requested sample
count—there is no hidden padding:

| Candidates | DDER step instances | Warm mean |
|---:|---:|---:|
| 256 | 8,960 | 22.48 ms |
| 512 | 17,920 | 26.12 ms |
| 1,024 | 35,840 | 36.27 ms |
| 2,048 | 71,680 | 47.95 ms |

Scaling is sublinear because the smaller workloads underfill the RTX 4080 and
all sizes retain fixed launch, winner-replay, and framework overhead.

## F. Equivalence results

### F.1 Complete rollout equivalence

Across nine frozen 35-step rollouts:

* maximum node-position difference: 2.265e-6 m;
* maximum node-velocity difference: 2.964e-5 m/s;
* maximum target-error difference: 1.043e-6 m;
* maximum directed-speed difference: 1.144e-5 m/s;
* maximum direction-error difference: 2.215e-4 deg;
* impact frame/time agreement: exact;
* valid/invalid agreement: exact;
* contact-order agreement: exact.

### F.2 Candidate ranking and MPPI update

For the frozen 2,048-candidate population:

| Metric | Result |
|---|---:|
| Maximum absolute cost difference | 1.526e-5 |
| RMS cost difference | 2.414e-6 |
| Pearson cost correlation | 0.999999999951 |
| Spearman rank correlation | 0.999999990221 |
| Top-1 agreement | yes |
| Top-5 overlap | 5/5 |
| Top-10 overlap | 10/10 |
| Valid classification agreement | 100% |
| Tip/non-tip classification agreement | 100% |
| Maximum importance-weight difference | 1.048e-8 |
| Maximum weighted-knot-update difference | 4.768e-7 m/s^2 |

### F.3 Closed-loop equivalence

The production receding controller was run with identical warm starts,
settings, disturbances, and seeds through separate unfused-reference and
accelerated processes.

Full distributed cable feedback:

| Condition | Reference success | Accelerated success | Accelerated mean error | Accelerated mean directed speed |
|---|---:|---:|---:|---:|
| Nominal | 3/3 | 3/3 | 4.71 mm | 3.93 m/s |
| Lateral sway | 3/3 | 3/3 | 9.39 mm | 3.85 m/s |
| Hidden interior velocity | 3/3 | 3/3 | 6.05 mm | 3.82 m/s |
| Before reversal | 2/3 | 2/3 | 27.60 mm | 3.78 m/s |
| After reversal | 2/3 | 2/3 | 11.11 mm | 3.45 m/s |

Across these 15 paired full-state runs, all success/failure classifications
agree. Maximum paired differences are:

* target error: 6.06e-6 m;
* directed speed: 1.34e-4 m/s;
* direction error: 0.00171 deg;
* impact time: 0 s;
* maximum drone displacement: 3.73e-5 m.

The accelerated hidden-interior-velocity endpoint-only ablation remains 0/3,
while full-state feedback is 3/3. This retains the main qualitative research
finding that distributed state contains control-relevant information.

One sensitivity must be reported. In a fresh paired reference run, one
endpoint-only seed landed marginally at the 3.5 m/s threshold (reference 1/3),
whereas the accelerated endpoint-only run was 0/3. The other two endpoint-only
seeds failed in both implementations, and all 15 full-state classifications
matched. Thus the implementation is control-equivalent over the full-state
test set but is not bitwise trajectory-equivalent under long, poorly observed,
threshold-sensitive endpoint-only receding execution.

## G. Remaining bottlenecks and Amdahl analysis

### G.1 Current ranking

1. **Fixed exact damping: 59.72 ms/update.** This remains the dominant target.
2. **Unchanged bending plus other fragmented step operations: 13.23 ms of
   exclusive device work.** The bending path is now worth profiling as a
   possible second exact-fusion target.
3. **Winner replay: 10.42 ms in the instrumented update.** It cannot be removed
   under the present batch-one prediction semantics.
4. **Projection: 2.26 ms.** Further work has little end-to-end leverage.
5. **Cost: 0.42 ms.** Further work is unjustified.

### G.2 Amdahl upper bounds

These are impossible best cases, not expected practical speedups:

| Component | Current wall fraction | Speedup if eliminated entirely |
|---|---:|---:|
| Fixed damping | 72.10% | 3.58× |
| All remaining other kernels | 15.97% | 1.19× |
| Projection | 2.72% | 1.028× |
| Fused cost | 0.51% | 1.005× |

Only a further 10.4% mean-time improvement is required to move from 82.84 ms
to 75 ms, but the p95 and system-integration margin matter more than the mean.

## H. Online feasibility

Using the final mean:

\[
f_{max}=1/0.082836 \approx 12.07\ \mathrm{Hz}.
\]

Using p95 gives approximately 11.69 Hz. The planner can therefore nominally
replan at 10 Hz on this GPU, but the stricter 70–80 ms budget has not been met.
No claim is made that a complete OptiTrack-to-flight-control loop will meet 10
Hz until sensing, state estimation, communication, and jitter are measured
together.

## I. Recommendation

Recommendation: **another exact optimization is still clearly justified.**

The next work should remain exact and should be decided by a focused final-step
profile:

1. optimize the fixed damping kernel further without changing float64,
   60-PCG, or its recurrence—for example measured layout/instruction-level
   changes, not speculative occupancy changes;
2. now profile and, if confirmed, fuse the unchanged bending-force path with
   adjacent integration work, because damping/projection fusion has made that
   formerly 1.3% path significant;
3. retain the batch-one winner replay unless controller output semantics are
   explicitly changed in a separate experiment.

Only after exact implementation work approaches diminishing returns should a
separate, explicitly authorized numerical-efficiency study evaluate fewer PCG
iterations. That study was not performed here.

## J. Unsuccessful or rejected changes

| Attempt | Result | Decision |
|---|---|---|
| Remove winner replay | About 440 ms at the early stage, but changed batch-one prediction numerics | Rejected |
| Three PCG accumulation streams/ILP | 91.44 ms mean in the later combined build | Reverted; slower |
| Four warps per damping block | 106.14 ms mean | Reverted; slower |
| NVRTC max-register count 64 | 89.85 ms mean with worse variance | Reverted; no benefit |
| Analytical bending force | Not attempted | Correctly deferred until reprofile |
| Fewer PCG iterations / float32 damping | Not attempted | Forbidden Phase-2 numerical change |

## K. Reproduction and artifacts

Primary source files:

* `cable_twin/shared/cuda_fixed_pcg.py` — fused damping and projection kernels;
* `cable_twin/shared/dder.py` — exact eligibility gates and generic fallback;
* `drone_mpc/simulator.py` — full-horizon CUDA graph;
* `drone_mpc/cuda_mppi_cost.py` — fused existing hard objective;
* `drone_mpc/mppi.py` — exact cost dispatch;
* `research_tools/freeze_mppi_reference.py` — frozen oracle generator;
* `research_tools/validate_mppi_optimization.py` — numerical/ranking validator;
* `research_tools/profile_mppi_forward.py` — staged CUDA-event profiler;
* `research_tools/profile_final_mppi_kernels.py` — final exclusive kernel profile;
* `research_tools/closed_loop_optimization_equivalence.py` — paired reference/optimized controller test.

Verification completed:

* frozen equivalence validation: passed;
* 189 Python unit tests: passed in 9.153 s;
* 20 warmed timings per retained stage: completed;
* full-state closed-loop paired seeds: completed;
* endpoint-only hidden-state ablation: completed.

All raw JSON measurements included in the delivery bundle are authoritative;
rounded values in this report are derived from those files.
