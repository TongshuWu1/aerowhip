# M0 → M1 → M2: system, adaptation and flight comparison

Review date: 10 September 2026. **Development evidence, not the future clean paper experiment.**

## Main assessment

The system is learning useful dynamics. M1 is a substantial improvement over M0.
M2 improves average prediction on the latest, previously untrained flight family,
but its benefit is smaller and not consistent across takes. Neither adaptation
has yet demonstrated reliable real-world target interception under the current
5 cm virtual-sphere definition. These are three model generations and **two
adaptation updates**, not three independent adaptation experiments.

The strongest new comparison uses all three frozen models on the same three M2
recordings, with identical commands, causal observations, time grids and masks:

| Model | Drone RMS, cm | Cable-tip RMS with measured attachment, cm | Complete command-to-tip RMS, cm | All-marker RMS, cm |
|---|---:|---:|---:|---:|
| M0 | 9.407 | 10.857 | 14.860 | 10.929 |
| M1 | 7.058 | 7.336 | 7.989 | 6.398 |
| M2 | 5.757 | 6.345 | 7.034 | 5.340 |

M2 reduces the mean complete tip error by **12.0% relative to M1** and **52.7%
relative to M0** on these three recordings. M2 take 002 nevertheless worsens
from 6.38 to 8.97 cm versus M1. All three models were frozen before these flights;
none trained on them. Calling 001/002 “future adaptation” does not retroactively
make them training data for M0/M1/M2. Their role matters if we later build M3.
These takes have now been inspected in development, so they are not a future
blind test.

![Matched model comparison](../../runs/evaluation/M0-M1-M2-system-review-20260910-v2/latest_paired_models.png)

All numbers are reproducible from the [checksummed review](../../runs/evaluation/M0-M1-M2-system-review-20260910-v2/report.json).
The [complete per-take tables](../../runs/evaluation/M0-M1-M2-system-review-20260910-v2/TABLES.md)
include training ancestry, exceptions and coverage. No models were fitted,
policies trained, MPPI jobs started or flight commands changed in this review.

## 1. Exact system and identities

The research question is whether a small amount of measured flight data can
improve a simulator sufficiently to plan better subsequent **open-loop aerial
whips**. Open-loop refers to the fixed PVA sequence during the maneuver. The
onboard tracking controller continues using feedback. Adaptation updates a
reusable prediction model between flights; it does not update the onboard
controller and does not directly add measured tracking errors to future commands.

```mermaid
flowchart LR
  P[Preliminary measured motions] --> M0[Initial model M0]
  M0 --> O0[Offline MPPI and frozen PVA / forecast]
  O0 --> F0[Real M0 flights]
  F0 --> ID1[Reviewed training takes + preliminary replay]
  ID1 --> M1[Adapted model M1]
  M1 --> O1[Offline MPPI and frozen PVA / forecast]
  O1 --> F1[Real M1 flights]
  F1 --> ID2[Reviewed training takes + prior training replay]
  ID2 --> M2[Adapted model M2]
  M2 --> O2[Offline MPPI and frozen PVA / forecast]
  O2 --> F2[Real M2 flights]
```

### Canonical flown lineage

| Paper/UI label | Exact catalog ID | Parent | Flown plan |
|---|---|---|---|
| M0 | `M0` | preliminary initialization/development | `20260910-022818-648386` |
| M1 | `M1-full` | M0 | `20260910-181929-716218` |
| M2 | `M2-frozen-refit-v1` | M1-full | `20260910-211435-608306` |

There are two other catalog entries that must not be drawn as additional stages:

- `M1` is the earlier **rejected gain-only sibling** of M1-full. It changed one
  drone gain and worsened validation. It is not the model used for M1 flights.
- `M2` is the original full fit. The separately named frozen-method refit
  reproduced its nominal parameters, both neural tensors and every reported
  per-take RMS exactly. It is a reproducibility check of the same generation,
  not an extra adaptation round or independent sample of performance.

The model signatures start `fc854eae6a37`, `024eb39ed29a` and `be4bd82c038d` for
the flown M0/M1/M2 respectively. Full signatures and referenced weight hashes
are in the review. M2 remains an unpromoted candidate in validation terminology;
the user's explicit selection to fly its CSV is a different decision.

### Physical and command interface

The repaired drone weighs 145 g and the cable/marker assembly 17 g. Individual
cable/marker masses are proportionally scaled engineering values, not newly
measured masses. Cable marker intervals total 0.9525 m; the first interval has
two computational segments and the other nine one each: 11 segments, 12 nodes,
10 moving observed cable sites plus the attachment. The marker mass is part of
the modeled assembly. Geometry and the tracked-origin-to-attachment body offset
`[0.006655, -0.012874, -0.055] m` remain fixed across these generations.

The selected start is tracked origin `[0,0,1.255] m`, target `[1.25,0,1.0] m`.
The planner outputs desired tracked-origin position, velocity and acceleration
at 30 Hz. Gravity is not added to the acceleration CSV. Piecewise-constant jerk
is integrated consistently at a command interval h:

\[
a_{k+1}=a_k+h j_k,\quad
v_{k+1}=v_k+h a_k+\tfrac12h^2j_k,\quad
p_{k+1}=p_k+h v_k+\tfrac12h^2a_k+\tfrac16h^3j_k.
\]

Those continuous reference equations and the zero-order-held transmitted packets
are different parts of the execution chain. The prediction uses the packet
schedule and effective delay. Logged rows match the frozen CSVs: 309, 311 and
310 rows per take in M0/M1/M2. This verifies logged command identity; it is not
independent proof of every onboard receipt timestamp or actuator response.

### What our simulator actually predicts

1. A loaded-drone response model maps delayed PVA commands to tracked pose.
2. Predicted attitude rotates the measured body offset to obtain attachment motion.
3. Attachment motion drives the discrete cable simulator and its optional residual.

The translational nominal response is

\[
\ddot p=K_p(p_d(t-\tau)-p)+K_d(v_d(t-\tau)-v)
       +K_{ff}a_d(t-\tau)+b_0+r_d(\cdot).
\]

XY/Z gains are tied within their respective axes. Initial effective compensation
and alignment are reconstructed from past observations; they are not measured
firmware integrator states. Attitude has a separate effective acceleration-to-
orientation response. The cable uses bending, damping and length projection,
with a 1/150 s outer step, eight internal substeps and four constraint iterations.
Its residual is an acceleration correction, not a learned target reward.

This is an **empirical loaded-drone → cable cascade**. Explicit simulated cable
tension is not fed back into the drone. Do not call it a bidirectionally coupled
first-principles UAV–rod model, a measured motor saturation model, or calibrated
battery compensation. The 2S battery/controller history is a collection covariate;
the current recordings do not isolate a battery effect.

## 2. What changed during fitting

### Initialization and the two updates

| Item | M0 | M1-full | M2 frozen refit |
|---|---|---|---|
| New observations | preliminary motions | M0 whip 001/002/004 | M1 whip 001/002/004 |
| Prior training replay | initial preliminary fit | preliminary training only | M0 training takes + preliminary training |
| Family weights | preliminary fit's own contract | new whip 2/3, preliminary 1/3 | new whip 1/2, prior whip 1/4, preliminary 1/4 |
| Drone | preliminary nominal + residual inherited | all effective gains/delay/attitude, then residual | same blocks warm-started and updated |
| Cable | reviewed development drag 0.4/s; NN disabled | EI, Cb, drag; newly initialized cable NN | EI, Cb, drag; existing cable NN updated |
| Validation | preliminary `figure8_002` | M0 005 usable; 003 reserved/history unusable | M1 003/005; prior/preliminary retention |

M0 is the selected preliminary **development** package. It is not simply the
unmodified original all-stage preliminary fit: cable damping and numerical
regularization were reviewed during development and the cable NN was disabled.
M1 also involved a reviewed continuation after an earlier cable gradient failure
and the removal of a 400-update neural ceiling. These facts prevent presenting
the entire historical sequence as three runs of an already unchanged recipe.
The subsequent M2 method freeze and reproducibility refit are valuable evidence
for the next clean study.

### Method name, objectives and numerical implementation

The method is **staged regularized nonlinear system identification by
simulation-error minimization**. It is conventional model fitting supporting
the system contribution, not a claimed new adaptation algorithm. See the
[frozen specification](../methods/FROZEN_SYSTEM_IDENTIFICATION.md).

For each stage, recursively simulate a whole window and minimize a robust
observation discrepancy plus parameter/residual regularization. In schematic form:

\[
L(\theta)=\sum_f w_f\frac1{|D_f|}\sum_{i\in D_f}
 \frac{\sum_t m_{it}\rho(\|\hat y_{it}(\theta)-y_{it}\|^2/s^2)}{\sum_t m_{it}}
 +\lambda_\theta\|\log\theta-\log\theta_{parent}\|^2+L_{NN},
\qquad \rho(z)=2(\sqrt{1+z}-1).
\]

Here the family/take/window weighting follows the saved loader; observations
are not treated as independent experiments. The expression summarizes the
stage-specific losses rather than asserting one joint objective over every block.

- **Physical/effective parameters:** SciPy bounded trust-region reflective
  nonlinear least squares, with positive parameters in log coordinates.
  Central finite-difference candidates are simulated in CUDA batches; the small
  outer trust-region solver runs on the CPU. This is not Adam on every parameter.
- **Drone delay:** profile the declared 0–120 ms candidate grid; choose using
  training loss plus a parent-delay prior, not held-out error. The grid is
  `[0,10,20,30,40,60,80,100,120] ms`.
- **Residual networks:** Adam, learning rate 0.001, weight decay 0.0001, gradient
  norm clipping 1, full recursive rollout gradients. Position/cable robust-loss
  scale is 2 cm; orientation scale is 0.05 rad. Parent-parameter prior weight
  is 0.03; residual magnitude/change regularizers use 0.01.
- **Order:** nominal drone translation/delay and attitude → drone NN → attitude
  refinement → cable physics → cable NN → freeze candidate → combined evaluation.
- **Cable loss:** half all-marker discrepancy and half tip discrepancy, driven
  by the **measured attachment**. This isolates the cable during identification.
  Complete command-driven tip error is evaluated afterward; it is not currently
  optimized as an extra combined stage.
- **Stopping:** retain the best verified checkpoint. Physical stages have
  plateau/solver criteria and an 80-function-evaluation ceiling; neural stages
  use practical plateau without a routine hard update ceiling. Report the actual
  stop reason rather than calling any exit “convergence.”

The ±0.5 m/s² per-axis residual bounds are regularization choices. They do not
establish hardware capability. Removing them would enlarge the function class
and change the frozen method; whether that helps requires a declared ablation.
The current evidence does not justify removing every constraint or increasing
training indefinitely.

### Measured parameter changes

| Effective parameter | M0 | M1 | M2 |
|---|---:|---:|---:|
| Position gain XY, s⁻² | 1.7214 | 1.5980 | 1.6470 |
| Position gain Z, s⁻² | 19.6683 | 16.7302 | 20.0305 |
| Velocity gain XY, s⁻¹ | 0.9955 | 0.8943 | 0.8431 |
| Velocity gain Z, s⁻¹ | 1.3581 | 1.6656 | 3.1990 |
| Acceleration feedforward XY | 0.7467 | 0.7283 | 0.7943 |
| Acceleration feedforward Z | 1.2990 | 1.7225 | 2.1533 |
| Effective delay, ms | 20 | 60 | 80 |
| Attitude time constant, ms | 81.45 | 40.43 | 32.47 |
| Attitude acceleration scale XY | 0.9909 | 0.9542 | 0.8897 |
| Attitude acceleration scale Z | 1.4982 | 0.6803 | 0.5400 |
| EI, N m² | 1.00001e-8 | 9.99729e-9 | 1.00243e-8 |
| Cb, N m² s | 9.99915e-5 | 9.91454e-5 | 1.00491e-4 |
| External cable damping, s⁻¹ | 0.40000 | 0.36601 | 0.28461 |

The changing Z response and attitude scales show why fitting the drone matters.
They do **not** mean the firmware gains changed or maximum thrust increased.
Effective delay also absorbs effects confounded with estimated clock alignment.
EI/Cb barely move while drag and the neural components change; this does not
prove accurate independent material identification. Excitation, sensitivity and
parameter correlations limit that claim. The fitting objective asks for useful
prediction, not uniquely recovered physical constants.

M1 selected drone residual used 585 total updates and cable residual 280. M2 used
260 and 200 respectively, retaining cable update 175. The original M2 fit took
27.8 minutes and the repeat 27.61 minutes on Windows/RTX 4080. More updates are
not inherently a better model. The same seed/contracts reproduced M2 exactly;
this establishes numerical repeatability for that experiment, not statistical
robustness across initializations or hardware.

## 3. Data treatment and fair evaluation

Raw world XYZ remains unchanged. Native OptiTrack data is approximately 100 Hz;
the controller's XYZ is cached OptiTrack data at a lower update rate, not an
independent position sensor. Their measured motions estimate the stream offset.
We do not align measurements to a desired trajectory or optimize clock offset
against a candidate model's prediction.

Missing markers remain missing. Geometry-only jump/chord rejection is shared
across models. Invalid rollouts fail evaluation rather than receiving convenient
candidate-specific masks. For causal diagnostics, the drone uses 0.4 s past
history and cable 1 s past history; cable velocity emphasizes recent samples
with a 0.02 s weighting constant. State initialization uses no future scored
samples. No in-window measured-state resets are used in the complete prediction.

M0 take 003 has insufficient strict cable history and stays out of reinitialized
model comparison. Its actual observed flight outcome remains in the prospective
table. It must not disappear from task-performance accounting. All three M2
takes passed the same strict initializer for this diagnostic. Intermediate
marker gaps remain masked. In the geometric approach window, M1 001/005 have
98.2%/96.4% contiguous-segment coverage; absence of an observed entry cannot
rule out an unobserved event inside a gap.

For a valid mask m and position error e, each take reports
`sqrt(sum(m * ||e||²) / sum(m))`. Generation summaries average those **per-take
RMS values equally**. They are not the pooled-frame RMS and not mean absolute
distance. The common model comparison evaluates its fixed 150 Hz grid from
time ≥0 to before the whip endpoint; the original-forecast reports retain their
original native sampling/scored intervals. Different metric conventions should
never be silently mixed.

Three complementary views are necessary:

| View | Inputs and initial state | Question answered |
|---|---|---|
| Original forecast vs real flight | exact preflight ghost, planned settled start | Did the deployed prediction describe the actual flight? |
| Same-command complete model | recorded packets; common causal physical observations, model-consistent hidden hover state | Which frozen model predicts this same flight better? |
| Conditional cable | same cable initialization; measured moving attachment | How much error remains in cable response with the boundary supplied? |

The second view is a postflight diagnostic, not a ghost that was available
before takeoff. Its difference from the first can include initialization,
recorded packet timing, sampling grid and observation masks. Do not attribute
that entire difference to initial cable motion. A clean initial-state ablation
must hold commands, clock, masks and grid fixed while changing only the
initializer; it remains a separate next experiment.

## 4. Prospective flights: what really happened

| Metric | M0, 5 takes | M1, 5 takes | M2, 3 takes |
|---|---:|---:|---:|
| Original drone RMS, cm | 12.280 | 8.456 | 7.041 |
| Original tip RMS, cm | 17.170 | 8.772 | 9.383 |
| Mean nearest target-centre distance, cm | 14.438 | 7.083 | 8.009 |
| Mean forward speed at nearest approach, m/s | 6.060 | 5.115 | 5.407 |
| Observed entry into 5 cm sphere | 0/5 | 0/5 | 0/3 |

![Prospective flight results](../../runs/evaluation/M0-M1-M2-system-review-20260910-v2/prospective_flights.png)

RMS covers each whip. Closest approach is computed on contiguous observed
segments through the whip plus 0.5 s of recovery, so a slightly late approach
is not discarded. This extended-window outcome is explicitly distinguished
from the planner's contact-terminated objective. Velocity is estimated with a
centered quadratic over five native samples (about 40 ms), with 3/7/9-sample
sensitivity retained. It is the forward component at nearest approach on a
miss, **not measured hitting power, force, impulse or successful-contact speed**.

M1 roughly halves original tip error and nearest distance versus the first
session. M2 further reduces average drone error but has slightly worse original
tip error and closest distance than M1. Its near-target speed is modestly higher.
This supports progress from the initial model, but not a monotonic task-success
story. All original forecasts predicted contact; no observed real entry occurred.
The user reported no physical cable contact or intervention during these takes.

The latest misses are not simply insufficient speed. At nearest approach, M2
takes 001 and 003 are approximately 8.0 and 7.7 cm to the negative-Y side of the
target, while take 002 is approximately 5.0 cm high and 4.25 cm short in X.
These are 3D targeting errors that an XZ-only replay can conceal. Increasing
forward speed alone would not establish a correction for those directions.

Measurement sensitivity also matters. The M2 intake's ±40 ms clock perturbation
changes equal-take original drone RMS over approximately 6.95–9.30 cm and tip RMS
over 9.38–14.14 cm. This is a sensitivity range, not a confidence interval or an
alternative fitted alignment. Three-to-nine-sample velocity windows place take
001's forward approach speed between roughly 4.99 and 6.08 m/s. Report the
estimator and sensitivity alongside any small speed improvement claim.

### The optimizer also changed

| Planning property | Flown M0 | Flown M1 | Flown M2 |
|---|---:|---:|---:|
| Samples / lookahead | 512 / 1.5 s | 512 / 1.5 s | 512 / 1.5 s |
| Committed whip duration | 1.1333 s | 1.2000 s | 1.1667 s |
| Full CSV with recovery | 10.2667 s | 10.3333 s | 10.3000 s |
| Contact before score in candidate ordering | historical score ordering | yes | yes |
| Extra successful-contact speed weight | absent | absent | 1600 |
| Reported search updates / time | 20 / 64.48 s | 31 / 102.35 s | 20 / 67.93 s |

The common search has ten support points, four timed proposal families, seed
657, target effective-sample-size fraction 0.2, minimum 20 iterations and plateau
patience 12. It is **offline MPPI-inspired exponential-weighted trajectory
optimization**, not live receding-horizon control or an exact importance-sampling
implementation of path-integral optimal control. A 1.5 s lookahead is not the
10 s recovery-inclusive flight length.

M2 adds `1600*v_forward²/(16+v_forward²)` at successful feasible contact; the
4 m/s scale is not a hard minimum. Increasing the preceding weight from 800 to
1600 retained exactly the same 35 commands and 4.905 m/s modeled contact speed.
The increased reward score was therefore not a new faster maneuver. All saved
seeds were editable proposals re-simulated in the current model. Reuse is valid,
but a paper must disclose it and compare against the seed itself.

Changing model, trajectory, ranking and speed reward confounds a simple
cross-generation task chart. The chart describes the development system as
flown; it cannot isolate adaptation's causal effect.

## 5. Controlled model comparisons and regression

### Earlier data retention

| Dataset and scope | Model | Drone RMS, cm | Conditional tip RMS, cm | Complete tip RMS, cm |
|---|---|---:|---:|---:|
| M0 take 005; excluded from all fits | M0 | 8.930 | 6.740 | 14.893 |
| same | M1 | 7.635 | 3.953 | 8.811 |
| same | M2 | 5.778 | 3.250 | 8.202 |
| M1 takes 003/005; excluded from all fits | M1 | 9.180 | 5.744 | 6.820 |
| same | M2 | 6.533 | 4.578 | 6.421 |
| All five M1 takes; includes M2 training | M1 | 8.742 | 5.632 | 5.846 |
| same | M2 | 5.967 | 4.416 | 6.232 |

The latest comparison reproduced **every M1/M2 per-take RMS in the prior frozen
report exactly**. Thus the previously observed regression is still present;
the improved average on new M2 flights does not erase it. On M1 take 005,
complete tip RMS worsens 4.97→6.99 cm, while on 003 it improves 8.67→5.86 cm.
On old M0 training take 004 it worsens 10.98→14.94 cm from M1 to M2.

Preliminary held-out mean-window drone/conditional-tip RMS changes from
7.80/4.92 cm under M1 to 8.09/5.04 cm under M2. These are preliminary-window
retention summaries from the earlier frozen audit, not additional whip takes.
The old rejected gain-only M1 also remains a negative result: M0 held-out 005
drone/tip RMS changed 8.93/14.89→10.41/17.93 cm.

### Latest M2 family: all three models were frozen before flight

| M2 take | M0 complete tip RMS, cm | M1 complete tip RMS, cm | M2 complete tip RMS, cm |
|---|---:|---:|---:|
| 001 | 16.509 | 8.712 | 6.672 |
| 002 | 10.018 | 6.380 | 8.966 |
| 003 | 18.054 | 8.876 | 5.464 |

The favorable average is supported by two takes, not all three. For take 002,
M2's drone RMS also worsens versus M1 (3.99→6.13 cm), while conditional cable
tip RMS improves (8.71→7.58 cm). This is direct evidence that a better conditional
cable model need not compensate for a worse predicted attachment trajectory.

![Per-take prediction traces](../../runs/evaluation/M0-M1-M2-system-review-20260910-v2/latest_prediction_traces.png)

### Why component improvement can coexist with a worse complete prediction

Let A be attachment trajectory, C the cable response and x0 the initial state.
A local error decomposition is

\[
e_{tip}\approx e_{cable\mid A}+J_A e_A+J_{x_0}e_{x_0}.
\]

This is an interpretation of nonlinear sensitivity, not a measured linear
identity. Squaring the sum creates cross terms. Two component errors can partly
cancel; reducing one component's norm can expose the other and increase total
error. Timing and direction matter, not just scalar RMS. The existing
[M2 regression analysis](M2_REGRESSION_ANALYSIS.md) used frozen component/axis
interventions and found that changed forward attachment motion contributed to
late tip-height regressions. It did not establish initial motion as the sole cause.

The current staged training optimizes drone motion and conditional cable motion,
then checks their composition. That is a defensible identification strategy with
an identifiable evaluation gap. It should be documented honestly. A proposed
combined loss, a robust initial-state planner and a new adaptation algorithm are
all distinct choices; none was silently implemented in this review.

### Are both residuals justified?

Both are active in M1/M2, but “both always improve performance” is unsupported.
Earlier fixed-parameter removal diagnostics found mixed effects: disabling the
M1 drone residual improved M0 005's complete tip error to 8.21 cm from 8.81 cm,
but worsened preliminary drone prediction. The cable residual improved M0 005
conditional tip error from 6.58 to 3.95 cm and complete tip error from 11.57 to
8.81 cm, with a small preliminary regression. Removing a component from a fitted
model measures local reliance; it is not the same as refitting an architecture
without that component. A paper claiming residual necessity needs the latter
comparison under matched data/budget, in addition to these inexpensive diagnostics.

## 6. Related work and what it means for this paper

The comparisons below use primary papers. They motivate evaluation choices;
their numerical thresholds and architectures are not automatically our protocol.
This targeted review does not establish a “first aerial whip” priority claim.

| Primary work | Relevant evidence | Consequence for our presentation |
|---|---|---|
| [Chi et al., Iterative Residual Policy, RSS 2022](https://arxiv.org/html/2203.00663v2) | Refines repeatable rope-whipping actions using observed trajectories and learned delta dynamics; evaluates distance versus attempts, varied ropes and robot embodiments, with system-identification and simulated-optimization baselines. | Closest conceptual comparator. Explain that we update an explicit reusable drone/cable forward model and replan, whereas IRP predicts action-induced trajectory changes. Show real trial budget, distance and failure cases, not only simulator loss. Their reported task accuracy is not directly comparable to our RMS. |
| [Chebotar et al., SimOpt, 2019](https://arxiv.org/html/1810.05687v4) | Updates simulation-parameter distributions from real/sim observation discrepancies and retrains policies; evaluates transfer, iterations and real-world trial counts against unchanged randomization. | Supports the iterative real–sim loop as established practice. Our point-model staged fitting and offline MPPI are different. Include a frozen-M0 baseline and disclose data/compute cost at each update. |
| [Mamedov et al., Learning DLO dynamics from a single trajectory, 2024](https://arxiv.org/html/2407.03476v1) | Uses regularized rollout fitting, explicit treatment of latent initial states, tests on other trajectories, and compares model accuracy and computational cost. Test initialization uses past observations. | Supports short-history state estimation and recursive prediction tests. Their jointly optimized training states and moving-horizon test estimator are not our weighted geometric initializer. Report initialization assumptions and out-of-training motion performance. |
| [DEFORM, Differentiable Discrete Elastic Rods, 2024](https://arxiv.org/html/2406.05931v2) | Combines differentiable rod modeling, parameter tuning, residual learning and constraints; evaluates rollout error, inference speed and component ablations using marker-observed cables. | Related to our hybrid cable model, not an exact implementation claim. Quantify physics/residual contributions and full-horizon behavior. Its conditional cable results do not validate our UAV-command-to-tip chain. |
| [Edraki et al., Human-Inspired Robot Whip Manipulation, 2025 workshop paper](https://deformable-workshop.github.io/icra2025/spotlight/01_01_05_Edraki_Human.pdf) | Compares preparation-plus-strike with strike-only, optimizing tip-distance and effort; uses a 1 cm geometric hit in its model and demonstrates robot transfer. | Separate task success from the preparatory mechanism. A simple primitive is a relevant planning baseline. Our 5 cm sphere and forward-fold preference are explicit project choices, not its validated thresholds. |
| [Williams et al., MPPI with covariance-variable importance sampling](https://arxiv.org/abs/1509.01149) | Formulates sampling-based stochastic control with path-integral weighting and parallel rollouts. | Cite the algorithmic inspiration, then specify our offline parameterized proposals, adaptive temperature, incumbent ordering and stopping. Do not claim its exact theoretical guarantees for our modified optimizer. |
| [Li et al., RotorTM, T-RO 2024](https://arxiv.org/abs/2205.05140) | Presents aerial manipulation simulation with payload connections and hybrid tension conditions, compared with real experiments. | Distinguish a free cable used for targeting from slung-load transport. Explain our effective loaded-drone cascade and its missing explicit reaction coupling. Aerial execution accuracy must be evaluated in addition to cable shape. |

The most useful lesson is to evaluate the **whole causal chain** with component
diagnostics alongside it. We do not need to invent another identification method
to justify a systems paper. We do need evidence that this particular combination
achieves a useful aerial manipulation capability at a reasonable real-data cost.

## 7. What reviewers are likely to ask

These are this review's recommendations, not universal rules stated by the papers.

| Reviewer question | Current answer | Evidence needed in the clean experiment |
|---|---|---|
| What is new beyond combining existing tools? | An intended aerial whipping system joining learned command response, cable prediction, offline planning and repeated measured refinement. | A precise capability claim, system design rationale and successful prospective experiments showing why the combination matters. Avoid unsupported “first” claims. |
| Does adaptation improve prediction, or did the reward/initial state change? | Matched-model diagnostics show average gains; original session comparisons are confounded. | Freeze reward, planner, initialization and evaluation; compare all models on identical excluded commands/observations. |
| Does better prediction actually improve targeting? | Initial improvement, then mixed task outcomes; no observed 5 cm entry. | Prospective target distance, entry counts, uncertainty/coverage and execution outcomes for a fixed target set. Do not substitute training loss for task benefit. |
| Is the result one lucky motion? | Three command families, 13 repeated takes, one target. | Several prespecified targets and repeats; disclose every attempted plan/flight and selection rule. |
| Is there train/test leakage? | Whole-take separation exists, but validation guided development. | Separate adaptation, operational validation and final untouched evaluation; freeze before final collection. Report ancestry, not just folder labels. |
| Why model the drone rather than prescribe perfect attachment motion? | Drone response changes materially; conditional cable error understates the complete task. | Same-command perfect/measured-attachment diagnostics and a drone-update ablation, with clear scope. |
| Are neural residuals necessary? | Mixed fixed-weight removal results. | Matched-data refits of nominal-only, drone residual, cable residual and full model if residual necessity is claimed. Otherwise modest claims and diagnostic evidence suffice. |
| How repeatable and efficient is the loop? | Exact M2 refit reproduction on one Windows/4080 setup. | Real-data duration/take budget, optimization time, search time, seeds, actual stops and all failures; separate offline time from flight time. |
| Is this truly a traveling whip wave? | The desired style is plausible; current bend-band gates are not a validated wave detector. | Ordered shape snapshots and curvature versus arc length/time with spacing and gaps explicit. Do not require an unvalidated detector to call geometric contact. |
| Are “power” and “hit” measured? | Only virtual geometry and tip kinematics. | Label virtual interception accordingly. Physical impact claims require independent contact/load measurement and a declared target. |

## 8. Recommended paper experiment and figures

### Primary claim and scope

A defensible target claim is: **a repeatable real-to-sim-to-real workflow improves
prediction and targeted open-loop aerial whipping using a learned loaded-drone/
cable model and offline sampling-based planning.** This is a claim to test, not
yet the conclusion of the current development recordings. The manuscript should
present system design and practical capability, with standard identification as
the supporting method. PPO is not necessary for this MPPI-focused experiment.

### Minimal controlled design

1. Finish development, then freeze the implemented identification recipe,
   target metric, planner reward/ranking, initialization and model selection rule.
   Choose either the current staged method or a deliberately tested extension;
   do not describe a proposed extension as implemented.
2. Build a fresh M0 from a declared preliminary dataset. At each round collect
   dedicated adaptation takes; use three repeats per training command as a
   practical starting design. Keep operational validation separate. Do not
   silently add validation takes to improve a fit.
3. Refit with the same documented family-weighting and stopping rule. Preserve
   every candidate, including regressions. A failed candidate remains a failed
   round; define beforehand whether the previous model is retained.
4. Generate each round's new MPPI plan with unchanged reward/search settings,
   same compute allowance and the same declared seed/proposal policy. Save both
   the starting proposal and optimized result. Previous commands may be warm
   starts if every round follows that same rule and their quality is reported.
5. After the method is frozen, collect a final comparison on at least three
   prespecified target positions, including a target absent from adaptation.
   Five repeats per target/model gives **45 final flights for three models**;
   this is a practical pilot design, not a claim of adequate statistical power.
   Additional repeats should be chosen from desired precision and observed
   between-flight variability before reading final outcomes.
6. In the final collection, interleave model-generated commands within target
   blocks where practical. Record battery voltage, controller/configuration,
   starting drone/cable motion, packet completeness and session order. This
   helps distinguish adaptation from battery/session drift. Keep the final
   measurements out of fitting and tuning.

The essential baseline is **frozen M0 with the same planner**. Evaluate all
frozen models on the same measured commands as a separate model-accuracy panel.
A simple preparation/strike primitive or unchanged initial command is a useful
second planning baseline. Do not add a large PPO campaign solely to manufacture
a comparator for this systems question. Expensive architecture ablations should
match actual claimed contributions rather than becoming a separate project.

A single M0→M1→M2 chain gives repeatability within one adaptation sequence.
Independent sequences or a second hardware/cable condition are needed for a
strong claim of general adaptation robustness. Forty-five repeats of planned
commands are not 45 independent learned models. Use per-take points and target-
level summaries; any uncertainty interval must respect the target/session/round
structure. Avoid frame-level significance tests and pooled “13 successes” logic.

### Main paper figures

| Figure/table | Content | Purpose |
|---|---|---|
| Fig. 1 | Hardware photograph, coordinate/attachment definition, real–sim loop | Define system boundary and contribution |
| Fig. 2 | Time-aligned desired drone, predicted drone, measured drone, predicted/observed cable | Show the full execution chain and actual whip |
| Fig. 3 | Same-flight M0/M1/M2 drone, conditional cable and complete tip error | Separate component identification from complete prediction |
| Fig. 4 | Prospective target distance and entry count versus cumulative real-data budget | Test the claimed benefit of adaptation |
| Fig. 5 | Several prespecified target layouts and per-take outcomes; one failure example | Show useful range, repeatability and limits |
| Table 1 | Hardware, masses, geometry, command/physics clocks and training split | Reproducibility |
| Table 2 | Frozen fitting/planning contract, data budget, actual compute and stopping | Cost and repeatability |
| Supplement | Residual/initial-state diagnostics, original ghosts, masks, clocks, parameter tables | Auditability without crowding the main result |

Do not put “M0 < M1 < M2” arrows on error plots as a guaranteed outcome. Show
individual takes, all-model counterfactuals and rejected variants in the
supplement. Use a task metric such as closest target distance as the primary
continuous outcome, and separately report the prespecified sphere-entry rate.
Keep 5 cm unchanged after observing misses; any alternative radius belongs in
a clearly labeled sensitivity curve, not a rewritten success claim.

## 9. UI and reproducibility delivered by this review

Open **Flight comparison → Study overview**. The three cards refer only to the
flown lineage; exact variant identities are available in tooltips and the model
details. Two modes separate original forecasts from the same-command model test.
The latter defaults to the latest M2 dataset and excludes any take used by any
compared model, including training ancestry. A separate selector permits an
explicit development view that includes training or reserved validation takes.
Dots show takes and thick marks equal take means. No page view starts fitting,
planning or evaluation.

The per-take table opens the **original flown ghost**, even when the selected
table row displays a different model's diagnostic error. The existing variants,
traces, fitting progress and recording browser remain available. The detailed
review is linked directly from the page. Missing or changed evidence clears
scores and displays an error instead of leaving stale numbers visible.

The new explicit audit performed 36 complete fixed-model take predictions
(4 usable M0 + 5 M1 + 3 M2, each under three models), plus conditional cable
diagnostics, in 139.27 s on Windows/RTX 4080. It verified 231 protected original
files unchanged. This is audit throughput, not MPPI latency. The first audit
preparation stopped before simulation on M1's empty original role mapping; its
manifest is preserved. The corrected run uses the later reviewed M2 preparation
roles, without changing historical protocols.

The UI data is [explicitly registered](../../config/evaluation/system_review.json)
against report and evidence hashes. Scientific figures are exported as PNG and
SVG alongside the machine-readable result and all-take tables. The diagnostic
source snapshot is preserved. Relevant checks include 19 focused tests and
native Windows Qt/VTK overview, minimum-window layout, exact-original-ghost
selection, playback and paired-actor verification. This does not establish
Ubuntu or other-GPU parity.

**Recommended next decision:** retain the current frozen models and use this
comparison to resolve initialization robustness and the remaining command-to-tip
gap before collecting the clean paper experiment. M2 has useful new-data
prediction gains; reliable real target interception remains the missing system
result. No additional fitting or planner change is implied by opening this review.
