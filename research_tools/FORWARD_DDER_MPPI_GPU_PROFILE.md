# Forward DDER–MPPI GPU profile

Date: 2026-08-26  
GPU: NVIDIA GeForce RTX 4080  
PyTorch/CUDA: 2.11.0+cu128 / CUDA 12.8

## Scope

This profile freezes the saved `11node_tru_phys.json` workload:

- 11 DDER nodes;
- 0.7 s horizon at 50 Hz: 35 physics steps;
- 11 three-dimensional acceleration knots;
- 2,048 candidates and two MPPI iterations;
- 4,096 sampled trajectories per update;
- one DDER substep;
- four position-projection iterations;
- one attached endpoint and one free endpoint;
- 60 fixed Jacobi-preconditioned float64 CG iterations per DDER step;
- matched controller/truth EI and Cb;
- gradient guidance disabled;
- existing one-step CUDA graph and chain Thomas solver unchanged.

The production controller and its physics were not modified. The primary
benchmark calls `optimize_mppi` directly. Internal attribution uses a separate
event-instrumented CUDA graph that evaluates the same equations. It was checked
against the production graph at batch 2,048 and batch one: maximum position and
velocity differences were exactly zero. A complete 35-step diagnostic rollout
also matched the production rollout exactly for every recorded tensor.

## End-to-end timing

Ten untouched warmed updates gave:

| Measurement | Time |
|---|---:|
| Cold first update | 1.110 s |
| Warm mean | 590.8 ms |
| Warm median | 590.3 ms |
| Warm minimum | 586.7 ms |
| Warm maximum / p95 | 597.8 ms |

The instrumented update took 593.6 ms. Its mutually exclusive attribution was:

| Update component | GPU-span time | Share |
|---|---:|---:|
| Two sampled DDER CUDA-graph rollouts | 423.6 ms | 71.4% |
| Final batch-one winner DDER replay | 146.4 ms | 24.7% |
| Drone integration + DDER input copies | 10.1 ms | 1.7% |
| Contact, safety, and cost evaluation | 7.9 ms | 1.3% |
| Noise, interpolation, weighting/selection, and unattributed host gaps | 4.3 ms | 0.7% |
| Candidate frame clones + trajectory stacking | 0.86 ms | 0.15% |
| **Accounted total** | **593.2 ms** | **99.93%** |

The entire captured DDER propagation therefore accounts for approximately
96.0% of the update. The final winner replay is unexpectedly important: it
reruns the selected control sequence at batch one after MPPI has already
simulated that candidate, costing nearly one quarter of the update.

## One-step DDER profile

The batch-2,048 profiling graph took a median 6.51 ms per physics step. The
batch-one graph took 3.99 ms. The small difference despite a 2,048× batch-size
difference is direct evidence that this workload is dominated by a very large
sequence of small kernels rather than arithmetic proportional to batch size.

For the batch-2,048 step:

| DDER component | Time/step | Share of profiled graph |
|---|---:|---:|
| Complete damping path | 4.576 ms | 70.3% |
| ├─ 60 PCG iterations | 3.029 ms | 46.5% |
| ├─ curvature-rate Jacobian | 1.362 ms | 20.9% |
| └─ system assembly, conversion, and reassembly | 0.185 ms | 2.8% |
| Position + velocity projection | 0.971 ms | 14.9% |
| Bending-force evaluation | 0.079 ms | 1.2% |
| Integration, clamping, event gaps, and other generated kernels | 0.885 ms | 13.6% |

For the batch-one step, damping was 68.4%, projection 23.6%, and bending 1.7%.
Weighting the batch-2,048 sampled rollouts and the batch-one winner rollout by
their measured production time attributes approximately:

- **67.0% of the whole MPC update to damping**;
- **44.2% to the 60 PCG iterations alone**;
- **20.5% to curvature-rate Jacobian construction**;
- **16.5% to position and velocity projection**;
- **1.3% to bending-force evaluation**.

These percentages use the numerically identical profiling graphs to subdivide
the measured production DDER time. External CUDA timing nodes add a small
profiling perturbation, so the untouched 590.3 ms median remains the
authoritative total.

## CG behavior

Production always executes all 60 iterations and performs no convergence test.
The separate residual diagnostic used

\[
r_{\mathrm{rel}} = \frac{\lVert Av-b\rVert}{\lVert b\rVert+\epsilon}.
\]

| Iteration | Median relative residual | p95 | Maximum |
|---:|---:|---:|---:|
| 1 | 5.20e-2 | 6.61e-2 | 7.25e-2 |
| 2 | 9.28e-3 | 1.18e-2 | 1.30e-2 |
| 5 | 4.06e-4 | 5.17e-4 | 5.67e-4 |
| 10 | 3.09e-8 | 3.94e-8 | 4.32e-8 |
| 20 | 3.45e-27 | 8.96e-27 | 1.84e-26 |
| 60 | 4.66e-101 | 9.51e-100 | 2.32e-98 |

This is strong evidence of numerical over-solving for these representative
states. It does **not** authorize changing the iteration count yet: doing so
requires an explicit trajectory, candidate-ranking, and closed-loop equivalence
study.

The production graph executes approximately 3,315 device operations per DDER
step, with a mean duration of only 1.75 µs. The dominant double-precision GEMV
kernel is called 60 times per step, matching PCG. The principal double reduction
kernel is called 121 times per step, matching the initial reduction plus two
reductions for every PCG iteration. At 35 steps, two sampled iterations, and one
winner replay, this is approximately 348,075 DDER device operations per MPC
update. The workload is therefore strongly small-kernel/dispatch dominated.

Nsight occupancy and memory-bandwidth counters were not claimed: the available
PyTorch trace identifies thousands of generated kernels, while a trustworthy
Nsight Compute study needs a separate targeted-kernel experiment.

## Projection

The existing chain-specific symmetric tridiagonal Thomas solver is being used.
Projection remains material because its tensor implementation launches many
small kernels:

- four position-projection assembly passes: 0.071 ms/step;
- four Thomas solves: 0.644 ms/step;
- four correction passes: 0.059 ms/step;
- velocity assembly: 0.022 ms/step;
- velocity Thomas solve: 0.161 ms/step;
- velocity correction: 0.014 ms/step.

The issue is not use of a dense solver. It is that a ten-equation chain solve is
expressed as a long series of general PyTorch tensor kernels.

## Trajectory storage and cost

The raw sampled cable positions occupy 9,732,096 bytes and velocities another
9,732,096 bytes, or 19.46 MB combined. This excludes drone/attachment arrays,
temporary stacking memory, cost intermediates, and allocator reservation.

However, candidate frame clones and trajectory stacking consume only 0.86 ms
per complete update (0.15%). Including post-rollout contact and cost evaluation
raises storage plus evaluation to 8.77 ms (1.48%). A streaming rollout remains
useful for memory and future fusion, but it is not the current runtime solution.

## Sample-count scaling

The diagnostic used one MPPI iteration and retained the same 35-step horizon
and final winner replay.

| Actual candidate batch | Warm median | Relative to B=256 |
|---:|---:|---:|
| 256 | 320.3 ms | 1.00× |
| 512 | 331.0 ms | 1.03× |
| 1,024 | 348.5 ms | 1.09× |
| 2,048 | 377.7 ms | 1.18× |

There is no hidden padding: each runtime uses the requested batch. Increasing
the real work eightfold increases time only 18%, because every batch executes
the same fixed sequence of thousands of small kernels and every solve also pays
for the constant batch-one winner replay. Conversely, simply reducing samples
cannot recover the desired update rate under the present implementation.

## Conclusions

### Top three bottlenecks

1. **Implicit damping: 67.0% of the update.** The fixed 60-iteration float64
   PCG and curvature-rate Jacobian together dominate.
2. **Projection: 16.5% of the update.** The chain solver is mathematically
   appropriate but fragmented into many small kernels.
3. **Redundant batch-one winner replay: 24.7% of the update.** This overlaps the
   damping/projection categories above but is a separate controller-level
   source of avoidable work.

Bending-force autograd is only 1.3% of the update. Replacing it analytically is
not the first performance change supported by this profile.

### Ranked exact implementation changes

**Largest plausible single change:** implement a fixed-topology fused damping
operator that preserves the current Jacobian, float64 arithmetic, Jacobi
preconditioner, and all 60 PCG iterations. Damping owns 67% of runtime and PCG
is expressed through hundreds of GEMV, reduction, comparison, and elementwise
kernels per step. This is the only single component with enough measured share
to produce a major factor improvement.

**Lowest-risk first engineering change:** retain the exact trajectory of the
best sampled candidate and remove its deterministic batch-one replay. This has
a directly measured 146 ms ceiling (24.7%) and changes neither physics nor
candidate selection. It should be implemented first as a correctness-preserving
baseline improvement even though it cannot reach the final target alone.

**Third change:** fuse the four position projections and velocity projection
for the fixed 11-node chain, preserving the existing Thomas mathematics and
iteration counts. Projection contributes 16.5% and is also launch dominated.

After those changes, capture or fuse the entire 35-step streaming rollout. Full
horizon capture by itself will reduce host replay overhead but will not remove
the thousands of kernels inside each DDER step. Analytical bending force and
streaming cost-only evaluation are lower-priority follow-ups based on the
measured shares.

The 70–80 ms goal is **not yet demonstrated achievable**. The profile does show
a plausible direction—remove a measured 146 ms replay and collapse the dominant
damping/projection kernel sequences—but only benchmarks of those exact changes
can establish whether the combined speedup is sufficient.

## Artifacts

- Raw JSON: `data/drone_mpc/profiles/11node_forward_profile.json`
- PyTorch CUDA trace: `data/drone_mpc/profiles/11node_forward_profile.torch_trace.json`
- Reproducible profiler: `research_tools/profile_mppi_forward.py`

Reproduce with:

```powershell
.\.venv\Scripts\python.exe -m research_tools.profile_mppi_forward `
  --warm-repeats 10 --sweep-repeats 5 --component-repeats 10
```
