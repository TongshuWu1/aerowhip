# Receding-horizon full-state DDER-MPPI: implementation and interim report

Date: 2026-08-25

## Executive result

The original one-shot forward-recoil whip was converted into a genuine
receding-horizon controller. Every solve now starts from the controller's
current drone and cable state, warm-starts by shifting the preceding control
sequence, predicts with the unchanged DDER cable model, executes only a short
prefix, and replans.

The main positive scientific result is conditional but clear. For a hidden
interior lateral-velocity perturbation that is exactly zero at both the
attachment and the free tip when injected, full-state DDER-MPPI succeeded in
3/3 trials (13.3 mm mean error), while endpoint-only DDER-MPPI succeeded in
0/3 trials (60.3 mm mean error). Thus, in this test, the distributed cable
state contains control-relevant information that instantaneous tip position
and velocity do not contain.

The benefit is not universal. A perturbation applied before reversal defeated
both controllers at the selected 64-sample, one-iteration budget, although
full-state feedback reduced mean error from 105.4 mm to 64.1 mm. The present
controller therefore demonstrates state-dependent recovery, not arbitrary
disturbance rejection.

The major engineering limitation is computation. A 64-sample, one-iteration
update takes about 10.4--10.8 s on the current CUDA implementation. None of the
requested 2--25 Hz replanning schedules meets its real-time deadline. All
closed-loop rate results below are zero-compute-delay simulation results.

## 1. Current physical and control pipeline

### 1.1 State

The predictive state is

\[
x_t = [p_d, v_d, r_{1:N}, \dot r_{1:N}], \qquad N=21,
\]

where `p_d,v_d` are drone translation and velocity and every DDER material
vertex has a 3-D position and velocity. The first cable vertex is attached to
the drone through a fixed 0.10 m vertical offset; the last vertex is free.

### 1.2 Model artifact used in this experiment

The controller loads the present 21-node, 0.961 m cable artifact:

- 11 measured material sites mapped to DDER nodes 0,2,...,20;
- 21 simulation vertices and 20 inextensible edges;
- measured nonuniform rest lengths;
- measured cable/marker mass distribution;
- `EI = 1.015478705e-4 N m^2`;
- `Cb = 1.5e-5 N m^2 s`;
- gravity `(0,0,-9.80665) m/s^2`;
- 3 DDER substeps per 0.01 s physics step;
- 4 inextensibility projection iterations per DDER substep.

This remains a **provisional free-tip transfer** from the latest two-holder
fit. The free-tip controller retains EI and Cb, removes GJ and both terminal
frame constraints, and adds one representative marker mass at the newly free
tip. Therefore torsion is not part of the present whip controller.

### 1.3 Drone propagation

The high-level input is commanded 3-D acceleration. The drone is currently a
translation-only double integrator:

\[
p_{d,k+1}=p_{d,k}+\Delta t\,v_{d,k}
             +\tfrac12\Delta t^2 u_k,
\qquad
v_{d,k+1}=v_{d,k}+\Delta t\,u_k.
\]

The acceleration vector is norm-bounded at 20 m/s^2. The safety objective
penalizes drone speed above 3 m/s. Cable forces do not feed back into the drone
translation model; a lower-level flight controller is assumed to track the
commanded acceleration.

### 1.4 DDER propagation

The cable is a straight, isotropic, inextensible discrete elastic rod. For
edge tangents `t_{i-1},t_i`, the exact DER curvature binormal is

\[
(\kappa b)_i =
\frac{2(t_{i-1}\times t_i)}{1+t_{i-1}^{T}t_i}.
\]

With zero intrinsic curvature, bending energy is

\[
E_b(r)=\frac12 EI\sum_i
\frac{\|(\kappa b)_i\|^2}{\bar\ell_i},
\]

and elastic force is obtained as `f_b=-dE_b/dr`. The fitted Cb enters an
objective Kelvin--Voigt curvature-rate dissipation. Runtime damping is solved
implicitly with the fixed-geometry corotational curvature-rate operator.

For each of the three DDER substeps:

1. interpolate the prescribed attachment position;
2. evaluate nonlinear elastic bending force;
3. add gravity and update the undamped velocity;
4. solve the implicit Kelvin--Voigt bending-damping velocity system;
5. advance vertex positions;
6. impose the moving attachment position;
7. apply four coupled inverse-mass edge-length projections;
8. project velocity onto the inextensible constraint tangent space.

Runtime rollouts use float32 CUDA and a captured fixed-shape DDER step. The
same equations are used by planner and plant in this matched-model study.

### 1.5 Prediction and control discretization

- predictive horizon: 2.0 s;
- physics rate: 100 Hz (`dt=0.01 s`);
- internal DDER update rate: 300 substeps/s;
- acceleration command rate: 50 Hz (`0.02 s`);
- optimization variables: 11 three-dimensional acceleration knots;
- knot interpolation: linear to the 100 command intervals;
- action dimension: 33 for this receding study;
- initial nominal sequence: saved successful forward-recoil whip;
- default disturbance-study replanning interval: 0.10 s.

### 1.6 MPPI sampling and update

At one MPPI iteration, antithetic Gaussian knot perturbations are generated
around the nominal sequence, with one exact nominal candidate. Perturbed
knots are vector-norm clipped, linearly interpolated, and every candidate is
rolled through the full 2.0 s DDER horizon from the current controller state.

For candidate cost `J_i`, MPPI uses

\[
w_i = \frac{\exp[-(J_i-J_{min})/\lambda]}
            {\sum_j\exp[-(J_j-J_{min})/\lambda]},
\qquad \lambda=1,
\]

and updates the nominal knots by the weighted applied perturbation. The best
sample is retained separately. For the selected online-budget experiment:

- 64 samples;
- one MPPI iteration;
- acceleration-noise sigma 2.5 m/s^2;
- pure MPPI (`gradient_guidance_fraction=0`).

No DDER control gradient is used in this receding-horizon experiment.

## 2. Exact unchanged strike objective

For every rollout, the event time is the first free-tip entry into the target
sphere if this occurs; otherwise it is the closest-approach time. In the
current implementation this is **geometric target contact**, not a rigid-body
collision event. The diagnostic named `physical_tip_contact` remains false.

Let `d` be tip-target distance, `v_parallel=v_tip^T d_hat`, and
`c=v_parallel/(||v_tip||+epsilon)` at the selected event. The task terms are:

\[
J_{pos}=40\frac{d^2}{d^2+0.18^2},
\]

\[
g(d)=\exp\left[-\frac{d^2}{2(0.12)^2}\right],
\]

\[
J_{speed}=25g(d)[3.5-v_{parallel}]_+^2,
\]

\[
J_{dir}=20g(d)[\cos(35^\circ)-c]_+^2.
\]

A valid strike requires all of:

- first geometric tip contact with `d <= 0.05 m`;
- directed speed at least 3.5 m/s;
- impact direction error at most 35 degrees;
- no safety violation.

A valid safe strike receives `J_success=-600`. Drone displacement at the
event is penalized by

\[
J_{disp}=2\|p_d(t_e)-p_d(0)\|^2.
\]

Safety penalties (total weight 180) retain the established checks for:

- excursion beyond 1.2 m from the original drone start;
- drone-target keepout below 0.3 m;
- drone speed above 3 m/s;
- ground/altitude violations;
- cable-drone clearance;
- a non-tip cable point entering the target before the tip;
- actuator-limit violation.

Weak acceleration effort and acceleration-change regularizers both use
weight `1e-5`. There is no energy, curvature, cable-shape, reversal, wind-up,
release, or manually specified whip term.

## 3. Receding-horizon implementation

The implemented loop is:

1. obtain the current plant drone and cable state;
2. initialize the first solve with the saved forward-recoil knots;
3. for later solves, shift the preceding optimized knots forward by the
   executed time and hold the final knot for the new tail;
4. run MPPI from the **current** controller state for a new 2.0 s DDER
   prediction;
5. keep displacement/excursion referenced to the original maneuver start;
6. execute only the first replan block;
7. independently advance the matched plant;
8. repeat until valid strike, invalid tip contact, non-tip-first event, safety
   failure, or 1.2 s timeout.

Every update saves its nominal/optimized knots, predicted drone trajectory,
predicted full cable positions and velocities, realized state, applied prefix,
cost, and planning time.

### Full-state feedback

The controller rollout starts directly from the actual current
`[r_1:N, rdot_1:N]` state.

### Endpoint-only ablation

The endpoint-only controller maintains its own DDER-predicted interior state.
At each update it receives only known drone state plus observed tip position
and velocity. If `s` is normalized material coordinate, it corrects its
predicted interior using

\[
w(s)=s^2(3-2s),
\]

so the correction is zero at the known attachment and equals the measured
error at the tip. The root is then clamped to the attachment. It never reads
true interior plant nodes.

## 4. Why multiple random seeds were used

MPPI is a stochastic sampling optimizer. A seed changes its Gaussian candidate
perturbations and can change whether a narrow whip basin is found. Five seeds
were used in the nominal compute-budget sweep and three seeds in comparisons.
This is a small reliability check, not an attempt to inflate the dataset.
After the user's request, the remaining trials were stopped; no additional
seeds are needed for this interim report.

## 5. Completed experiments and results

### 5.1 Sample/iteration efficiency sweep

Each cell contains five nominal matched-state trials. Every configuration
succeeded. Key results are:

| Samples | Iterations | Mean error (mm) | Directed speed (m/s) | Median update time (s) | Rollouts/strike |
|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 18.47 | 6.26 | 10.35 | 64 |
| 32 | 2 | 13.27 | 6.18 | 15.27 | 128 |
| 32 | 4 | 12.24 | 6.22 | 25.08 | 256 |
| 64 | 1 | 11.85 | 6.20 | 10.44 | 128 |
| 64 | 2 | 11.23 | 6.25 | 15.44 | 256 |
| 64 | 4 | 12.58 | 6.32 | 25.42 | 512 |
| 128 | 1 | 11.16 | 6.21 | 10.54 | 256 |
| 128 | 2 | 11.50 | 6.26 | 15.67 | 512 |
| 128 | 4 | 11.29 | 6.46 | 25.99 | 1024 |
| 256 | 1 | 11.36 | 6.21 | 15.62 | 512 |
| 256 | 2 | 12.05 | 6.13 | 26.32 | 1024 |
| 256 | 4 | 13.82 | 6.33 | 46.60 | 2048 |

Interpretation: 64 samples and one iteration is the compute/quality knee.
Increasing sample count beyond 64 or repeatedly updating the sampling mean
does not monotonically improve hard strike quality. Warm-start quality is doing
most of the work. The apparent 128 rollouts/strike at 64x1 comes from two 2 Hz
replans in this sweep, not 128 samples in one update.

### 5.2 Replanning-rate study

Three seeds were used per requested rate with 64 samples and one iteration.

| Requested rate | Success | Error (mm) | Directed speed (m/s) | Direction error (deg) | Median update time (s) | Deadline met |
|---:|---:|---:|---:|---:|---:|---:|
| 2 Hz | 3/3 | 11.92 | 6.06 | 32.24 | 10.41 | 0% |
| 5 Hz | 3/3 | 12.31 | 6.47 | 27.14 | 10.64 | 0% |
| 10 Hz | 3/3 | 17.24 | 5.23 | 29.92 | 10.55 | 0% |
| 25 Hz | 3/3 | 20.86 | 7.50 | 22.81 | 10.77 | 0% |

The closest control-aligned rate to the requested 20 Hz was 25 Hz because the
50 Hz command grid requires the replan interval to be a multiple of 0.02 s.

Interpretation: the lowest tested rate, 2 Hz, is already sufficient in the
zero-delay matched-state simulation. Higher rates do not monotonically improve
quality because every stochastic replan can modify an already good nominal
trajectory. Computationally, none is online-capable: even 2 Hz needs a 0.5 s
deadline but takes about 10.4 s per update.

### 5.3 Open-loop versus closed-loop disturbances

The table reports the trials completed before the requested stop.

| Perturbation | Controller | Valid strikes | Mean error (mm) | Directed speed (m/s) | Mean direction error (deg) |
|---|---|---:|---:|---:|---:|
| unperturbed | open loop | 1/1 | 22.7 | 6.23 | 31.8 |
| unperturbed | full state | 3/3 | 17.2 | 5.23 | 29.9 |
| unperturbed | endpoint only | 3/3 | 17.2 | 5.23 | 29.9 |
| +30 mm forward sway | open loop | 1/1 | 35.3 | 6.11 | 32.8 |
| +30 mm forward sway | full state | 3/3 | 16.9 | 5.20 | 31.3 |
| +30 mm forward sway | endpoint only | 3/3 | 16.9 | 5.20 | 31.3 |
| -30 mm backward sway | open loop | 1/1 | 22.2 | 6.29 | 31.0 |
| -30 mm backward sway | full state | 3/3 | 15.9 | 6.20 | 31.8 |
| -30 mm backward sway | endpoint only | 3/3 | 15.9 | 6.20 | 31.8 |
| 30 mm lateral sway | open loop | 0/1 | 59.6 | 6.39 | 26.5 |
| 30 mm lateral sway | full state | 3/3 | 30.6 | 5.44 | 21.3 |
| 30 mm lateral sway | endpoint only | 3/3 | 30.6 | 5.44 | 21.3 |
| hidden interior lateral velocity | open loop | 1/1 | 38.1 | 6.37 | 26.7 |
| hidden interior lateral velocity | full state | **3/3** | **13.3** | **5.83** | **18.6** |
| hidden interior lateral velocity | endpoint only | **0/3** | **60.3** | **4.21** | **41.4** |
| hidden velocity before reversal | open loop | 0/1 | 95.7 | 5.49 | 44.9 |
| hidden velocity before reversal | full state | 0/3 | 64.1 | 4.15 | 55.4 |
| hidden velocity before reversal | endpoint only | 0/3 | 105.4 | 5.23 | 37.1 |
| hidden longitudinal velocity after reversal | open loop | 0/1 | 44.7 | 4.80 | 48.7 |
| hidden longitudinal velocity after reversal | full state | **3/3** | **19.6** | **4.52** | **32.3** |

The requested stop occurred before endpoint-only after-reversal trials, distal
lash trials, the separate static same-tip pair sweep, and goal generalization.
Those results are deliberately not inferred.

## 6. What the experiments show

### Finding 1: receding-horizon execution is functionally correct

The controller repeatedly replans from the current propagated state, uses a
shifted previous solution, and executes only a prefix. Nominal strikes remain
valid down to the lowest tested 2 Hz zero-delay update rate.

### Finding 2: warm starts make a very small MPPI budget sufficient nominally

All efficiency cells succeeded because the first solve begins near a known
forward-recoil solution. One iteration and 64 samples are enough for nominal
local refinement; this result must not be interpreted as cold-start maneuver
discovery.

### Finding 3: feedback itself repairs endpoint-visible sway

Open-loop lateral sway misses by 59.6 mm, while both closed-loop controllers
recover valid strikes around 30.6 mm. Full and endpoint-only results match
exactly for these low-order sway cases because the imposed sway profile is the
same smooth root-to-tip mode assumed by the endpoint observer. This is evidence
for closed-loop replanning, not for a distributed-state advantage.

### Finding 4: distributed state is useful for hidden interior velocity

For the interior `sin(pi s)` lateral velocity perturbation:

- root velocity perturbation: zero;
- tip velocity perturbation: zero;
- tip position perturbation: zero;
- interior velocity-state L2 difference: 1.2649 m/s over the nodes.

At the first update, full-state and endpoint-only therefore receive identical
drone/tip observations, while only full-state sees the interior motion. The
initial nominal sample remains selected in both controllers, but their knot
sequences diverge strongly from the second replan onward (approximately
12--15 m/s^2 L2 knot difference at update 2, growing thereafter). Across equal
MPPI seeds, full state converts this condition to 3/3 valid strikes; endpoint
only produces two timeouts and one invalid tip contact.

This is direct evidence that instantaneous tip state is not Markov for this
whipping task and that a distributed cable estimate can improve control.

### Finding 5: state information does not remove optimizer/phase limits

The before-reversal disturbance remains 0/3 even with full state. Full state
reduces mean error by about 39%, but the hard strike still fails. One full-state
trial reaches 16.6 mm at 4.58 m/s yet fails because its direction error is
53.1 degrees, outside the 35-degree cone. A distance-only evaluation would
incorrectly label it successful.

### Finding 6: current replanning is not computationally practical online

The observed 10.4--10.8 s update time is over 20 times a 2 Hz deadline and over
100 times a 10 Hz deadline. The study currently answers a control-information
question under zero compute delay. It does not yet demonstrate a deployable
real-time controller.

### Finding 7: repeated replanning reduces clearance margin in nominal runs

The nominal open-loop non-tip margin beyond the 5 cm target sphere is about
29.7 mm. Nominal 10 Hz closed-loop runs average only about 7.5 mm. They remain
valid under the current logic, but this is a robustness warning: stochastic
replanning can improve target error while moving other cable sections closer
to the target.

## 7. Main failure modes and limitations

1. **Runtime:** sequential 2 s DDER rollouts dominate the online budget.
2. **Provisional plant:** EI/Cb are transferred from a two-holder artifact;
   free-tip identification is not yet definitive.
3. **Matched-model assumption:** planner and plant use exactly the same DDER
   model and parameters.
4. **Translation-only drone:** attitude, thrust dynamics, cable reaction force,
   and low-level tracking error are omitted.
5. **Geometric contact:** target contact is a 5 cm distance event, not a rigid
   collision model.
6. **Small seed count:** three comparison seeds support a pilot conclusion,
   not a final statistical claim.
7. **Phase sensitivity:** a hidden disturbance before reversal remains outside
   the recoverable basin at 64 samples and one iteration.
8. **Clearance:** some closed-loop solutions have small non-tip margin.
9. **Incomplete requested sweep:** distal-lash, dedicated same-tip pair, and
   goal-generalization studies were stopped at the user's request.

## 8. Recommended current configuration and next decision

For continued simulated controller experiments, use:

- 2.0 s prediction horizon;
- 64 samples;
- one MPPI iteration;
- shifted forward-recoil warm start;
- 5--10 Hz simulated replanning for phase-resolution studies;
- full distributed cable state when available.

For a claimed online controller, **no current configuration is recommended**:
the implementation misses every tested real-time deadline. The next research
decision should be whether to optimize/parallelize the forward DDER rollout or
use a reduced control-rate/horizon representation while preserving the same
physics and hard task objective. That computational work should be evaluated
separately from changes to rewards or adaptation.

## 9. Answer to the main research question

Current evidence supports the following precise statement:

> In matched-model simulation, shifted-warm-start receding-horizon DDER-MPPI
> can preserve and locally correct a dynamic whip using a small stochastic
> search budget. Full distributed cable feedback materially improves recovery
> for interior velocity disturbances that are not observable from the
> instantaneous tip state, but it does not guarantee recovery for all
> disturbance phases. The control formulation is currently computationally
> impractical for online execution because each MPPI update takes roughly
> 10.5 seconds.

This is stronger and more defensible than claiming either that full state is
always necessary or that the present controller is already real time.
