# Event-triggered distributed-state DDER adaptation

Date: 2026-08-26  
Status: clean simulated-state study; no sensing noise, dropout, latency, or unmodeled physics

## Executive result

An asynchronous two-parameter adaptation layer was added around the existing
accelerated 11-node DDER--MPPI controller without changing its physics,
numerical method, objective, contact logic, safety logic, or warm-start logic.
The estimator uses short measured-root DDER rollouts, central finite
differences in log parameter coordinates, a 2-by-2 Levenberg--Marquardt update,
held-out validation, and atomic model publication.

Across the prescribed nine-case truth matrix:

- the matched cable triggered zero fits and had zero parameter drift;
- every mismatched cable triggered a useful first fit;
- the mean two-parameter log error fell from 0.314 initially to 0.131 after
  one strike and 0.0558 after repeated three-strike adaptation;
- adapted all-node position RMSE at 0.1, 0.2, and 0.3 s was 0.0195, 0.0324,
  and 0.0793 mm, versus nominal-model errors of 0.111, 0.237, and 0.351 mm;
- full event triggering reduced first-strike fitting attempts from 72 to 8
  (89%) relative to continuous fitting;
- in the 10-paired-seed closed-loop study over mismatched cables, fixed
  nominal MPPI succeeded in 66/80 trials, one-strike adaptation in 69/80,
  three-strike adaptation in 72/80, and the oracle model in 72/80;
- three-strike adaptation repaired six nominal failures and introduced zero
  paired failures, matching oracle success classification on this test set.

This establishes the required causal chain for the clean study:

\[
\text{parameter correction}
\rightarrow
\text{distributed prediction correction}
\rightarrow
\text{improved strike reliability}.
\]

The target-error and speed distributions do not improve monotonically for
every seed. That is expected because MPPI is stochastic and the fixed nominal
controller already succeeds in a broad portion of this task. The defensible
control result is recovery of mismatch-induced failures, not a claim that
oracle physics minimizes every finite-sample trajectory metric.

## A. Architecture

The implementation has two timing domains.

### Fast control loop

At the beginning of each MPPI update, the controller reads one immutable model
generation and freezes it for the entire solve. It then runs the unchanged
accelerated 11-node DDER--MPPI planner and executes one prefix. Parameter
objects cannot be modified during a solve.

### Slow adaptation loop

The estimator executes:

\[
\text{monitor}
\rightarrow
\text{buffer}
\rightarrow
\text{trigger}
\rightarrow
\text{select}
\rightarrow
\text{fit}
\rightarrow
\text{validate}
\rightarrow
\text{publish}.
\]

Fitting is available synchronously for replay and between-strike operation,
and through a single-worker asynchronous interface for concurrency tests. The
controller never waits for the worker.

The plant truth and controller estimate are separate objects. The estimator
receives distributed positions, velocities, and measured attachment motion,
but never receives truth parameters.

## B. Parameterization and preserved physics

Only

\[
\eta_E=\log(EI/EI_0),\qquad
\eta_C=\log(C_b/C_{b,0})
\]

are estimated, using

\[
EI_0=1.0154787051679222\times10^{-4}\ \mathrm{N,m^2},\qquad
C_{b,0}=1.5\times10^{-5}\ \mathrm{N,m^2s}.
\]

The 11-node grid, material coordinates, vertex masses, rest lengths, gravity,
bending energy, curvature definition, Kelvin--Voigt damping equation,
float64 fixed-60 PCG damping solve, one DDER substep, four position
projections, velocity projection, and attachment treatment are unchanged.

The finite-difference prediction path calls the same accelerated forward DDER
operators used by MPPI. No autograd control gradient, generic optimizer,
learned residual, or parameter refit of other quantities was introduced.

## C. Buffer design

- Recent FIFO duration: 2.0 s.
- Informative cache: at most eight 0.1 s measured segments.
- Maximum observed memory in this study: 90.2 KiB.
- Candidate window: newest 0.3 s plus cached segments.
- Multiple-shooting segment: 0.1 s, initialized from the measured cable state.
- Selected per fitting event: four fitting segments and two chronological
  held-out validation segments.

Segments containing target contact, safety failure, invalid observations, or
numerical invalidity are excluded. Selection combines excitation and temporal
diversity so that a later hover cannot evict every reversal/lash segment.

## D. Trigger logic

The cheap health prediction uses a 0.1 s distributed DDER rollout every 0.1 s.
The metric is mean squared error over the ten dynamic cable vertices. Its EMA
uses alpha 0.90.

The exact-state thresholds are globally fixed at:

\[
E_{\rm high}=10^{-9}\ \mathrm{m^2},\qquad
E_{\rm low}=2.5\times10^{-10}\ \mathrm{m^2}.
\]

The matched numerical floor was approximately \(10^{-14}\ \mathrm{m^2}\),
whereas the smallest prescribed mismatch produced approximately
\(9\times10^{-9}\ \mathrm{m^2}\). These thresholds are valid only for the
clean exact-state study and must be recalibrated once, globally, for real
measurement noise.

The mismatch must persist for 0.2 s. Hysteresis requires recovery below the
low threshold before a new error trigger, and cooldown is 1.0 s.

Excitation pre-gating uses normalized relative cable velocity, curvature, and
curvature rate. The information gate then requires:

\[
\lambda_{\min}(J^TJ)>10^{-7},\qquad
\kappa(J^TJ)<10^7.
\]

The cheap excitation score is explicitly a prefilter, not a Fisher
information claim.

## E. Efficient fitting

For each selected segment and LM iteration, five hypotheses are propagated in
one segment-by-hypothesis batch:

\[
\eta,\quad \eta\pm0.03e_E,\quad \eta\pm0.03e_C.
\]

All dynamic node-position residuals are normalized by cable length. Central
finite differences produce the two residual-Jacobian columns. The update is:

\[
\Delta\eta
=-(J^TJ+10^{-4}I)^{-1}J^Te.
\]

The 2-by-2 system is solved directly. Log-space trust regions limit one step
to \(\log 1.2\) in EI and \(\log1.3\) in Cb. Absolute ratios are bounded by
[0.5, 2.0] and [0.4, 2.5]. Line-search factors 1, 0.5, 0.25, and 0.125 are
evaluated as one held-out batch. At most two LM iterations occur per trigger.

Median accepted-fit time was 76.5 ms, p95 82.9 ms. A typical accepted event
used 72 short segment-hypothesis rollouts. The dense information solve itself
is negligible; short DDER propagation is the fitting cost.

### Finite-difference validation

Log perturbations from 0.005 to 0.08 were tested. On stroke, reversal, and
lash segments, sensitivity directions agreed with the chosen 0.03 direction
at cosine at least 0.996 (usually above 0.9999). The selected step is therefore
inside a stable central-difference range for this clean simulation.

## F. Observability and phase analysis

For the representative truth ratio (0.8, 0.7):

| Phase | lambda_min | Condition | EI/Cb sensitivity correlation | Gate |
|---|---:|---:|---:|---|
| Early stroke | 1.93e-7 | 15.3 | +0.379 | pass |
| Reversal | 7.44e-6 | 1.99 | +0.279 | pass |
| Distal lash | 3.75e-6 | 29.1 | -0.933 | pass, highly correlated |
| Weak late motion | 5.42e-8 | 8.06 | -0.139 | reject |

The reversal is the best joint-identification phase in this maneuver. Distal
lash has large sensitivity but a narrow correlated valley, so using it alone
would be risky. The 17-by-17 loss landscapes for all four phases placed their
sampled minimum at EI ratio 0.800 and Cb ratio 0.688; the Cb difference from
0.700 is the grid resolution.

## G. Parameter convergence

The truth matrix was:

\[
(1,1),(0.8,1),(1.2,1),(1,0.7),(1,1.3),
(0.8,0.7),(1.2,1.3),(0.8,1.3),(1.2,0.7).
\]

After three retained strikes, estimates were:

| True EI | True Cb | Estimated EI | Estimated Cb |
|---:|---:|---:|---:|
| 1.0 | 1.0 | 1.0000 | 1.0000 |
| 0.8 | 1.0 | 0.8551 | 1.0305 |
| 1.2 | 1.0 | 1.1538 | 0.9745 |
| 1.0 | 0.7 | 1.0246 | 0.7323 |
| 1.0 | 1.3 | 0.9691 | 1.2529 |
| 0.8 | 0.7 | 0.8368 | 0.7474 |
| 1.2 | 1.3 | 1.1481 | 1.2509 |
| 0.8 | 1.3 | 0.8213 | 1.2826 |
| 1.2 | 0.7 | 1.2226 | 0.7403 |

Across mismatched cases, mean log-parameter error was 0.314 initially, 0.131
after one strike, and 0.0558 after three strikes: reductions of 58.3% and
82.2%, respectively.

The matched condition generated zero triggers, zero fits, and zero drift.

## H. Prediction improvement

Mean errors over mismatched cases and three dynamically different start times:

| Horizon | Model | All-node position RMSE | All-node velocity RMSE |
|---:|---|---:|---:|
| 0.1 s | nominal | 0.111 mm | 0.00206 m/s |
| 0.1 s | adapted | 0.0195 mm | 0.000266 m/s |
| 0.1 s | oracle | 0.000035 mm | 0.00000077 m/s |
| 0.2 s | nominal | 0.237 mm | 0.00429 m/s |
| 0.2 s | adapted | 0.0324 mm | 0.000811 m/s |
| 0.2 s | oracle | 0.000061 mm | 0.00000107 m/s |
| 0.3 s | nominal | 0.351 mm | 0.00462 m/s |
| 0.3 s | adapted | 0.0793 mm | 0.000764 m/s |
| 0.3 s | oracle | 0.000085 mm | 0.00000126 m/s |

The nonzero oracle floor is float32 runtime propagation and output conversion,
not model mismatch.

## I. Adaptation efficiency

First-strike scheduling over all nine cases:

| Policy | Fit attempts | Accepted | Short DDER rollouts | Fitting time |
|---|---:|---:|---:|---:|
| Continuous | 72 | 20 | 3296 | 4.11 s |
| Error-triggered | 8 | 8 | 576 | 0.666 s |
| Full event-triggered | 8 | 8 | 576 | 0.666 s |

The full gate and error-only trigger are equal here because every natural whip
window selected after a true error was informative. This does not make the
information gate redundant: the phase diagnostic shows that weak late motion
fails it. Continuous fitting performed eight rejected fits on the matched
cable alone.

## J. Parameter ablations

Mean final log-parameter error over the eight mismatched cases:

| Adaptation variables | Mean error |
|---|---:|
| EI only | 0.274 |
| Cb only | 0.176 |
| Joint EI + Cb | 0.0558 |
| Joint without information gate, informative data only | 0.0558 |

Single-parameter adaptation can compensate in the wrong physical parameter.
For example, with true (EI,Cb)=(1.0,0.7), EI-only fitting moved EI to 1.154
while Cb remained wrong. Joint fitting is retained.

## K. Closed-loop control

Ten paired MPPI seeds were run for every truth condition and every controller
model. Across the 80 mismatched trials:

| Controller | Success | Wilson 95% interval | Median target error | Median directed speed |
|---|---:|---:|---:|---:|
| Fixed nominal | 66/80 (82.5%) | 72.7--89.3% | 8.38 mm | 3.65 m/s |
| Adapted after strike 1 | 69/80 (86.3%) | 77.0--92.1% | 8.48 mm | 3.61 m/s |
| Adapted after strike 3 | 72/80 (90.0%) | 81.5--94.8% | 8.45 mm | 3.63 m/s |
| Oracle | 72/80 (90.0%) | 81.5--94.8% | 8.18 mm | 3.65 m/s |

Paired fixed-to-three-strike outcomes were:

- both succeed: 66;
- nominal fails and adapted succeeds: 6;
- nominal succeeds and adapted fails: 0;
- both fail: 8.

After one strike there were five repairs and two degradations. This supports
bounded repeated adaptation rather than publishing an aggressive single jump.

Including the matched condition, results were 75/90 fixed, 78/90 after one
strike, and 81/90 after three strikes/oracle. One seed failed for every model
in the matched condition, demonstrating an MPPI discovery failure independent
of parameter adaptation.

## L. Publication semantics and timing

The accelerated full-horizon CUDA graph captures EI and Cb tensors when a
`WhipSimulator` is built. Therefore, changing a Python estimate in place is
incorrect. The implementation constructs a complete immutable snapshot and
simulator for an accepted generation, prewarms required graph shapes outside
the current MPPI solve, and atomically swaps the entire runtime.

Validation at truth ratio (0.8,0.7) found:

- adapted published runtime versus independently constructed truth: exactly
  zero position and velocity difference;
- stale nominal runtime versus truth: maximum 3.10 mm position difference and
  0.0478 m/s velocity difference;
- one batch-one graph rebuild/prewarm: 0.416 s;
- 2048-candidate plus batch-one full graph rebuild/prewarm: 0.621 s.

The graph rebuild cannot be on the 10 Hz critical path.

## M. Concurrent fitting interference

Twenty warmed MPPI updates were compared with and without same-GPU fitting:

| Timing | MPPI only | MPPI + concurrent fit |
|---|---:|---:|
| Median | 82.9 ms | 88.2 ms |
| p95 | 85.6 ms | 92.2 ms |
| Maximum | 85.8 ms | 93.7 ms |

Concurrent fitting increased median MPPI latency by 6.4% and p95 by 7.7%.
The fitting job itself slowed to a 165 ms median under contention. Although
planning remained under 100 ms in this desktop test, 92 ms leaves inadequate
margin for OptiTrack, state estimation, communication, and jitter. The first
physical deployment should therefore use between-strike fitting and off-path
runtime prewarming. Concurrent fitting remains a later scheduling experiment.

## N. Failure modes and limitations

- EI and Cb are correlated in some phases, especially distal lash.
- Weak recent motion is not informative even when an old model error exists.
- One bounded fitting event is deliberately incomplete; repeated interactions
  improve convergence.
- Better physical prediction does not monotonically improve every stochastic
  MPPI trajectory metric.
- Current thresholds assume exact state and cannot be transferred directly to
  OptiTrack data.
- The plant differs only in EI and Cb. Mass, length, drag, attachment,
  actuator, contact, and sensing mismatch remain untested.
- The controller study reuses the same nominal warm-start family; robustness
  across targets and maneuver families remains separate.

## O. Scientific decisions

1. **Is EI observable from natural whip motion?** Yes, particularly around
   reversal and lash; early stroke is weaker but usable.
2. **Is Cb observable?** Yes in the clean distributed-state experiment.
3. **Can both be identified simultaneously?** Yes; joint fitting is much more
   accurate than either single-parameter ablation.
4. **Most informative phase?** Reversal for well-conditioned joint inference.
5. **Does the informative cache help?** Mechanistically yes: it retains
   reversal/lash data while the latest weak-motion window fails the information
   gate. A dedicated cache-off statistical ablation remains useful before a
   paper claim.
6. **How many fits are avoided?** 64 of 72 first-strike attempts (89%) versus
   continuous fitting.
7. **Does adapted DDER predict better?** Yes, by factors of approximately
   4.4--7.3 in all-node position RMSE over 0.1--0.3 s.
8. **Does prediction improve control?** It improved mismatched success from
   82.5% to 90.0% after three interactions, matching the oracle success count.
9. **Same strike or subsequent strikes?** The bounded estimator is mainly a
   next-strike method; one strike helps, three are more reliable.
10. **Does concurrent fitting interfere?** Yes modestly; it consumes most of
    the remaining 10 Hz timing margin.
11. **EI-only or joint?** Joint EI+Cb is retained.
12. **Is active excitation justified now?** No. Natural reversal supplies
    strong information. Active excitation should be considered only after the
    real sensing study shows repeated information-gate failures.
13. **Paper contribution status?** Promising in clean simulation, but not yet
    sufficient alone. The next decisive evidence is causal OptiTrack-state
    adaptation with noise/dropout/latency and a cache-on/cache-off ablation.

## Reproduction

Core implementation:

- `drone_mpc/distributed_adaptation.py`

Experiments:

- `research_tools/distributed_adaptation_study.py`
- `research_tools/adaptation_identifiability_diagnostics.py`
- `research_tools/adaptation_mppi_interference.py`
- `research_tools/validate_adaptation_publication.py`

Primary data:

- `data/drone_mpc/adaptation/distributed_adaptation_policy_matrix_v1.json`
- `data/drone_mpc/adaptation/distributed_adaptation_ablation_matrix_v1.json`
- `data/drone_mpc/adaptation/distributed_adaptation_control_10seeds_v1.json`
- `data/drone_mpc/adaptation/adaptation_identifiability_diagnostics_v1.json`
- `data/drone_mpc/adaptation/adaptation_mppi_interference_v1.json`
- `data/drone_mpc/adaptation/adaptation_publication_validation_prewarm_v1.json`

