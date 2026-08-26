# Exact acceleration of the DDER endpoint-history observer

## Outcome

The moving-history observer was accelerated without changing its observation,
state, physical model, correction basis, optimization, or acceptance rule.

For the production observer workload on the RTX 4080:

| Quantity | Frozen implementation | Accelerated implementation |
|---|---:|---:|
| DDER nodes | 11 | 11 |
| History | 0.30 s / 16 frames | 0.30 s / 16 frames |
| Correction variables | 24 | 24 |
| Central-FD candidates | 49 per iteration | 49 per iteration |
| GN/LM iterations | 2 | 2 |
| Line-search candidates | 4 | 4 |
| Mean active update | 248.5 ms | **24.94 ms** |
| Median active update | not frozen | **24.59 ms** |
| p95 active update | 266.8 ms | **26.70 ms** |
| Measured mean speedup | — | **9.96x** |

The accelerated observer now has enough steady-state margin for a 10 Hz
observer schedule by itself.  MPPI remains a separate sequential cost in the
complete controller, so this number is not a claim that the entire sensing,
observer, MPPI, and communication stack is already below 100 ms.

The one-time CUDA graph construction takes approximately **326 ms** and peaks
at **24.67 MiB** allocated memory in the benchmark.  The GUI pays this cost in
an explicit prewarm phase before live execution, not during the first active
history correction.

Machine-readable timings are saved in
`reports/figure8_history_observer_data/runtime_optimized.json`.

## Preserved method

The accelerated code still uses exactly:

- root position and root velocity plus free-tip **position** history;
- no tip-velocity observation;
- no distributed plant-state observation;
- no future measurement or reference-path input;
- the same recursive 11-node DDER state;
- the same four spatial correction modes;
- the same 24 position/velocity correction coefficients;
- the same position and velocity correction scales;
- the same inextensibility projections;
- the same fixed DDER step, EI, Cb, mass, gravity, damping, PCG, and projection;
- the same central finite differences;
- the same normalized tip-position residual;
- the same SVD diagnostic and numerical-rank threshold;
- the same GN/LM normal equations;
- the same prior, trust region, component bounds, and line search;
- the same two-iteration limit and acceptance threshold.

No variable was removed, no candidate was pruned, no time step was skipped,
and no lower-fidelity model was introduced.

## Changes retained

### 1. Complete fixed-history CUDA capture

Previously, Python replayed one already-captured DDER physics step for every
history frame.  The accelerated operator captures the complete fixed-shape
operation:

1. construct the smooth position and velocity correction;
2. project the corrected initial state;
3. propagate all 15 DDER transitions under measured root motion;
4. stack the 16 predicted states.

Separate graphs are cached for the 49-candidate central-difference batch and
the four-candidate line-search batch.  Each graph accepts new corrections,
anchor state, measured root history, and time intervals through static input
buffers.  The numerical operation order inside one trajectory is unchanged.

### 2. GPU-resident cable histories

The former implementation copied every candidate's full position and velocity
trajectory to NumPy, then copied the accepted trajectory back to CUDA.

The accelerated implementation keeps full histories on CUDA.  Only the small
free-tip position history needed by the unchanged NumPy GN/LM calculation is
transferred to the CPU.  Accepted states are published by device-to-device
cloning.

### 3. Cached immutable and measured tensors

Immutable DDER runtime constants are no longer rebuilt at every recursive
step.  Root positions and velocities are converted once when ingested, and
the complete root/time history is materialized once per correction rather
than once per candidate replay.

### 4. Removed exact redundant evaluations

The former update performed two unnecessary physical evaluations:

- a standalone nominal rollout immediately before a finite-difference batch
  whose first member was the same nominal control vector;
- a batch-one replay after line search even though the selected line-search
  trajectory had already been propagated.

The accelerated implementation uses candidate zero of the central-difference
batch as the nominal evaluation and retains the selected line-search state on
the GPU.  This does not change the objective, line-search choice, correction,
or published state.

### 5. Removed repeated global synchronization

The former replay explicitly synchronized CUDA before and after every batch.
The new implementation uses CUDA events for device timing.  The one required
tip-history copy naturally establishes the dependency needed by the CPU
linear algebra, without an additional pre-replay global barrier.

### 6. Explicit prewarm

The Figure-8 worker prewarms both fixed graph shapes before entering the live
controller.  Graph construction remains visible as setup time and is not
hidden inside steady-state timing.

### 7. Captured recursive propagation

The 50 Hz batch-one propagation used between optimization updates now has its
own fixed CUDA graph.  Its measured mean wall time fell from **1.63 ms** to
**0.35 ms** (4.68x) while retaining the complete DDER step.  Graph-owned output
buffers are cloned on device before being appended to the immutable history.

## Numerical and behavioral validation

Three validation levels were added or rerun:

1. **Former Python replay versus fixed-history replay on CPU:** full positions,
   velocities, and tip histories are bitwise equal for deterministic batched
   correction candidates.
2. **Captured CUDA replay versus eager execution of the same fixed operator:**
   full positions and velocities are bitwise equal.
3. **Captured recursive batch-one step versus eager DDER:** positions are
   bitwise equal and the largest observed velocity difference is
   `1.92e-11 m/s`, far below float32 physical resolution.
4. **Repository regression:** all **201** tests pass, including the original
   hidden-interior-velocity observer test and Figure-8 observation-mode tests.

The deterministic active benchmark still reduces the tip-history measurement
RMSE from **17.310 mm** to **5.037 mm**, with the same 24-dimensional correction
and two GN/LM iterations.  This benchmark is for runtime and implementation
equivalence; the frozen scientific control results remain in
`FIGURE8_DDER_HISTORY_OBSERVER_REPORT.md`.

## Current timing anatomy

For 20 repeated active updates of the same hidden-state correction case:

| Component | Mean | Approximate update share |
|---|---:|---:|
| Two 49-candidate FD graph replays | 11.30 ms | 45.3% |
| Two four-candidate line-search graph replays | 11.25 ms | 45.1% |
| SVD, GN/LM, copies, bookkeeping | 2.39 ms | 9.6% |
| Total | **24.94 ms** | 100% |

The remaining runtime is now mostly the required physical work: four complete
DDER history-batch evaluations for an accepted two-iteration update.  Removing
one of those evaluations would change the estimator iteration or line-search
method.  The small CPU solve is no longer a meaningful bottleneck.

## Research interpretation

This is an implementation acceleration, not a new estimation method and not a
standalone paper contribution.  It supports the research claim by allowing the
same transparent physics-based observer to operate much closer to the intended
online rate without weakening the DDER model or hiding computation through a
reduced observer.

The appropriate next end-to-end measurement is the sequential latency of:

`measurement -> history correction -> 1024-sample/2-iteration MPPI -> command`

under the current GUI configuration.  That measurement should include p95 and
maximum latency before claiming a complete 10 Hz real-time controller.
