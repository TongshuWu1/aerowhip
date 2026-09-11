# Research design: identification and adaptation for aerial cable strikes

Proposal and development evidence, 2026-09-05. This document separates a proposed
conference experiment from implemented software and preliminary results. It does
not claim physical parameter recovery, reliable real flight, or established novelty.
It complements `DIFFERENTIABLE_IDENTIFICATION.md` and `PAPER_EXPERIMENT_PROTOCOL.md`.

**The central question.** Can a small budget of instrumented real strike trials
improve a physically grounded simulator enough to improve *new* open-loop aerial
cable strikes from measured initial conditions?

The first study should use one cable configuration and one policy algorithm.
Generalization across initial cable motion and target locations is already a
substantial question. Additional cables, changing material properties, and PPO/SAC
comparisons should extend a convincing result rather than multiply an unresolved
identification problem.

**1. Define what is known, estimated, and learned.**

| Quantity | Meaning | Proposed treatment |
| --- | --- | --- |
| Geometry, marker masses, drone mass, coordinate transforms | Properties measured independently of strike outcomes | Calibrate and version; propagate measurement uncertainty |
| Physical coefficients theta, initially EI and Cb | Shared coefficients of the chosen cable model | Estimate from informative experiments; report uncertainty and model assumptions |
| Initial state z0 for each trial | Drone and cable positions and velocities at handover | Estimate causally from pre-strike observations |
| Actuator parameters phi | Command-to-realized-force response, delay, saturation | Identify separately from cable response |
| Neural weights psi | Remaining repeatable motion discrepancy | Learn after physical fitting; do not interpret as material properties |

The ten cable markers observe the centerline at ten points. They do not directly
measure material twist or every deformation between markers. Our twelve-node model
adds the attachment and one interpolated node in the first interval. Calling this
the complete *discretized model state* is appropriate only after estimating
velocities and stating the omitted twist/internal-state assumptions.

Better prediction after changing EI does not establish that EI is closer to the
true material value. It may be an effective coefficient compensating for a boundary
condition, discretization, air motion, or state-estimation error. For an unchanged
cable, later fitting updates our estimate; it does not imply the material changed.

**2. Initialization is part of the method.**

First distinguish initial *model calibration* from initial *state estimation*.
They happen at different times and must have separate validation.

For calibration, verify marker identity, units, axes, quaternion convention,
timestamps, segment arc lengths, and the actual attachment constraint. Keep the
user-described top-of-drone tracking origin and 55 mm downward attachment offset
as the current geometry hypothesis. Validate that transform independently; do not
let EI silently correct it. Also document the tracked origin relative to the
center of mass. A rotating attachment offset contributes to attachment velocity.
Measured chords between markers need not equal cable arc lengths.

For each trial, use a short history ending at the actual force-mode handover to
estimate positions and velocities. Reconstruct a length-consistent centerline
with measurement-noise weighting and a modest smoothness prior for unobserved
nodes. Report how much reconstruction moves each measured marker. Enforce
velocity compatibility with the length constraints and moving attachment. Check
whether marker offsets from the physical centerline matter at the required scale.

The existing offline centered velocity smoother uses future samples around a
window start. That can support an offline reference estimate, but it is not a
deployable causal initializer. Report both an offline-reference evaluation and a
deployment-matched evaluation; never silently substitute the former for the latter.
Freeze the estimator before comparing adaptation methods. Any state refinement
against an entire observed strike is a retrospective diagnostic, not an initial
condition that the deployed controller could have known.

Repeat nominally identical trials to estimate initial-state variation and trajectory
repeatability. Perturb initial positions and velocities within measured uncertainty
in simulation. This establishes how much prediction error could plausibly come
from initialization before attributing it to material coefficients.

**3. Collect experiments that distinguish physical effects.**

| Experiment family | Purpose | Important limitation |
| --- | --- | --- |
| Static bending under known geometry/loading | Constrain bending stiffness independently of damping | A straight vertical hanging cable may provide very little stiffness information |
| Released deflections and decaying oscillations, varied amplitudes and planes | Constrain dynamic response and damping | Decay can also contain aerodynamic and attachment losses |
| Prescribed attachment sweeps, including existing figure-eight recordings | Validate dynamic response and excitation coverage | Gentle motion may not cover strike speeds or curvature |
| Held-out strike-like attachment motions | Test task-relevant prediction before policy adaptation | Must remain separate from the fitting data |

The generalized-coordinate DER identification paper uses dedicated static-shape
and twist-buckling experiments, with separate scalar searches for stiffness
quantities. This supports separating experimental effects; our proposed damping
tests are an additional project design, not that paper's protocol.
[Chen, Bretl and Pham](https://arxiv.org/html/2310.00911v3).

Measure the attachment behavior: free pivot, clamped tangent, or compliant joint.
Test candidate boundary models only when physically plausible. A point pivot
cannot be made into a clamp just by adjusting EI. Use the measured attachment
trajectory when fitting cable dynamics, so drone tracking error is not attributed
to the cable. A clamp may additionally require measured orientation.

Do not choose a universal recording count in advance of a pilot. Start with
repeated independent takes in each relevant motion family, then use parameter
sensitivity and between-take variation to decide what additional excitation is
needed. More nearly identical windows mainly increase computation.

**4. Fit physical coefficients and test whether they are identifiable.**

For trial j, let y be measured marker positions, b the attachment boundary motion,
H the mapping from simulator nodes to observed markers, and E the fixed initializer:

\[
\hat z^j_0=E(y^j_{[-T_{pre},0]}),\qquad
\hat z^j_{t+1}=F_{\theta}(\hat z^j_t,b^j_{t:t+1}).
\]

Fit positive bounded parameters using recursively predicted trajectories:

\[
\theta^*=\arg\min_\theta\;
\sum_j w_j\frac{1}{N_j}\sum_{t,i}
\rho\!\left(\|H_i\hat z^j_t-y^j_{t,i}\|_{\Sigma_i^{-1}}\right)
+\lambda_\theta R(\theta).
\]

Here N normalizes valid observations, weights balance takes/motion families,
Sigma represents measured observation uncertainty, and rho is a robust loss.
These covariance weights and priors are proposed extensions; the current fitter
uses a robust position loss without this full uncertainty model. Treat timestamp
or initial-state nuisance variables with independently justified bounds if they
are estimated at all. Unrestricted per-trial nuisance fits can hide a wrong model.

Use broad initialization and multiple starting points. With only two coefficients,
a log-parameter loss surface and a derivative-free reference are practical and
informative. Compare solvers using the same objective and simulation budget.
Differentiability is a computational tool, not evidence of a correct fit.
Planar Robot Casting calibrates its simulator using Differential Evolution before
policy learning, illustrating a successful alternative workflow.
[Lim et al.](https://arxiv.org/abs/2111.04814).

Inspect a profile loss: fix EI at successive values and re-optimize Cb, and vice
versa. Check sensitivity to measurement noise, initialization, time step, and
discretization. Resample whole independent takes for uncertainty estimates.
A broad valley, boundary solution, or large changes across reasonable assumptions
means the coefficient is weakly determined. Report a supported range or collect
more informative data instead of printing a highly precise point estimate.

Shorter training windows can help optimization, but final evaluation must make
uninterrupted predictions over the actual strike duration. No measured interior
cable state may reset the prediction after initialization. Measured future boundary
motion is permitted in this *conditional cable-identification* experiment and must
be clearly distinguished from full command-driven flight prediction.

**5. Add a residual only after evaluating physical-only performance.**

Freeze the selected physical coefficients and learn a small correction inside the
simulator transition. Our current correction is bounded cable-node acceleration,
applied before damping and length projection. Train using predicted states and
multi-step trajectory loss; the network must not see future measured cable states.
Regularize correction magnitude over rollout states and evaluate whether it injects
unreasonable energy or deteriorates outside the training motion range. The current
implementation penalizes correction magnitude at initial states only, so this
broader regularization remains proposed.

DEFORM combines differentiable DER, learned integration correction, and constraint
handling. Its main two-end manipulation assumptions differ from our one-attachment
system; our implementation is not an exact reproduction.
[Chen et al., DEFORM](https://arxiv.org/html/2406.05931v3).

Residual forces learned from sparse markers are established in soft robotics.
[Gao et al.](https://arxiv.org/abs/2402.01086).
The March 2026 RAFL preprint also learns residual acceleration fields and examines
transfer across geometries. Thus an additive acceleration network alone is not a
defensible novelty claim.
[Cho and Chen, RAFL](https://arxiv.org/abs/2603.22039).

The present MLP is translation invariant but does not enforce rotation equivariance
or momentum conservation. Do not claim either. If generalization across strike
directions fails, test local geometric features or shared node/edge networks as a
controlled architectural comparison. A bounded output is not a stability proof.

**6. Use real attempts to diagnose and update the right component.**

Open-loop execution can still be fully recorded. The policy receives the initial
state, compiles a force sequence, and executes it once to a planned cutoff before
PID recovery. Passive MoCap logging during the strike supports adaptation after
the trial and does not introduce within-strike feedback. Low-level attitude
stabilization must be described separately from the open-loop high-level maneuver.

Record both successes and failures: drone pose, all visible cable markers,
timestamps and quality flags, commands, mode transitions, target geometry,
initial-state estimate, model/policy versions, and trial outcome. Log available
actuator telemetry. Existing cmd_full_state reference messages are not direct
measurements of realized force. Reconstructed force from differentiated motion is
also an estimate with uncertainty, not a thrust sensor reading.

| Diagnostic replay | Interpretation if it predicts poorly |
| --- | --- |
| Measured attachment motion + estimated initial cable state | Cable model, boundary model, or initialization needs investigation |
| Actual commands + estimated initial drone/cable state | Full-system mismatch; compare with the first replay to investigate force tracking and delay |
| Same replay over plausible initial-state perturbations | Quantifies sensitivity to sensing and handover uncertainty |

These comparisons narrow causes; they do not uniquely prove a cause. Fit a
command-to-force/vehicle response model separately if the prescribed-boundary
cable prediction is adequate but command-driven flight prediction is poor.
The offset transform, actuator lag, and downwash must not all be represented as
changes in cable stiffness.

Exclude or explicitly model target contact when fitting free-flight cable
dynamics. Define the contact truncation and tracking-quality rules before looking
at which method wins. A failed attempt is useful because its trajectory exposes
mismatch, not because the binary failure label identifies EI or Cb. Keep successful
attempts to constrain already-correct behavior. Compare failure-only selection with
all-attempt adaptation if failure selection is a claimed contribution.

**7. Adapt between trials without forgetting or test leakage.**

Maintain preliminary data D0 and a growing adaptation buffer Dk. One proposed
physical update minimizes a weighted mixture of their trajectory losses plus a
scaled prior/trust-region penalty around the previous estimate. Choose mixture
weights and penalty strength using development data. Preserve motion-family
coverage so many near-identical failures cannot overwhelm the baseline data.

Re-estimate physical coefficients with the residual disabled, anchored by the
physical experiments. If coefficients remain poorly identifiable, retain their
supported range rather than chase every trial. Refit the residual against the new
physical model; compare zero initialization and warm starting on development data.
Never carry an old correction across a changed physical model without reevaluation.
Joint optimization of physics and residual can be an ablation, but it makes their
individual interpretation harder because either can compensate for the other.

Freeze each candidate model and evaluate its predictions before retraining the
policy. Then use a controlled policy-training budget and test on fresh real trials.
This separates a better simulator from a luckier policy run. SimOpt already updates
simulation parameter distributions using real rollouts interleaved with policy
training; the general adaptation loop is established prior work.
[Chebotar et al.](https://arxiv.org/abs/1810.05687).

Split whole takes or sessions before making windows. Maintain development
validation, adaptation trials, and a protected final test. Once a trial enters
adaptation, it cannot serve as unseen evidence for that updated model. For an
online learning curve, save each prediction before observing the next trial, then
score it; separately reserve final trials never used for selection or updating.
Report the real-trial cost of tuning and evaluation as well as adaptation.

**8. Evidence needed for a conference claim.**

The proposed hypotheses are: deployment-matched initialization reduces prediction
error; physical fitting improves unseen trajectories; a residual improves further
without unacceptable loss of generalization; and real-data adaptation improves
new-trial prediction and real strike performance at a fixed data budget.

Compare a frozen preliminary model, physical-only adaptation, residual-only
adaptation with fixed physics, and staged physical-plus-residual adaptation.
Use identical data for offline fitting comparisons. For real adaptation loops,
report each method's actual collected data and interaction budget; diverging
policies do not collect identical trajectories. Add simple parameterized-action
optimization or a direct between-trial action-adaptation baseline if claiming
that model updating is the best route to successful strikes.

IRP is especially relevant: it uses observed rope trajectories to refine actions
between attempts. It addresses rope whipping without requiring the same physical
parameter-update mechanism. Compare transfer to new initial states/targets, not
only repeated correction of one target.
[Chi et al., RSS 2022](https://www.roboticsproceedings.org/rss18/p016.pdf).

| Proposed paper figure | What it establishes |
| --- | --- |
| Initial reconstruction and velocity uncertainty | What information was available at deployment |
| EI/Cb profile loss and independent-take estimates | How strongly the data constrain physical coefficients |
| Marker and tip error versus prediction lead time | Recursive predictive accuracy on the same held-out starting windows |
| Tip trajectory, speed, direction and planned cutoff | Task-relevant error hidden by whole-cable averages |
| New-trial prediction error and strike success versus real adaptation attempts | Whether more real data improves transfer |
| Physical/residual adaptation ablations | Which component causes the improvement |

Use physical units, raw exported data, PDF/SVG figures, and uncertainty intervals
resampling independent takes/sessions or adaptation runs rather than individual
100 Hz frames. State the aggregation, sample counts, failures, exclusions, and
which outcomes have missing impact measurements. Use common starting windows for
lead-time plots. Report tip error and high-error quantiles alongside mean marker
RMSE. Position accuracy alone does not establish valid hit speed or direction.

Treat the existing 5 cm hit radius as a task definition, not a generic acceptable
RMSE threshold. A low averaged RMSE can still hide a missed target or timing error.
Use independently scored real hit and recovery rates to establish task performance.
Freeze target and initial-state distributions and repeatability procedures before
the main comparison; size the final experiment from pilot variability and the
precision of the improvement claim, not an arbitrary episode count.

**9. What the current diagnostic actually shows.**

The completed job `20260905-061423-970034-fit` refines the residual from an earlier
development fit while keeping its physical coefficients frozen. The selected
coefficients are approximately EI = 2.84e-8 N m² and Cb = 3.75e-5 N m² s. They are
development estimates, not independently verified material constants.

| Validation window duration | Active marker RMSE | Fitted physics | Physics + residual | Hybrid tip RMSE |
| --- | ---: | ---: | ---: | ---: |
| 1 s | 75.36 mm | 74.47 mm | 71.75 mm | 115.93 mm |
| 2 s | 97.41 mm | 91.35 mm | 89.06 mm | 151.33 mm |
| 5 s | 123.07 mm | 107.34 mm | 104.94 mm | 177.63 mm |
| 7 s | 166.05 mm | 134.18 mm | 130.98 mm | 223.85 mm |

These are arithmetic means of per-take RMSE on two validation recordings, with
measured attachment motion supplied. Within each horizon, the methods share
windows. Different horizons can use different starts and numbers of valid windows;
this table is not a common-cohort error-growth curve. It has no confidence intervals
and is not a real strike result. The protected take remains unused.

The residual improves these aggregate errors modestly. Physical refinement changed
the training objective very little after the broad grid initialization, and
backpropagation produced extremely large pre-clipping gradients. These observations
motivate checking conditioning and identifiability; they do not identify the cause
of the mismatch. The candidate has not been applied automatically.

**10. Next work, in order.**

1. Freeze this diagnostic and define the measurement/state/actuation contract.
2. Implement and validate causal initialization; quantify repeatability and error
   sensitivity using the existing recordings and new independent repeats.
3. Evaluate boundary hypotheses and EI/Cb profile losses; collect targeted bending
   and decay experiments where the existing motions are uninformative.
4. Compare physical-only and residual models on held-out strike-like motion with
   identical starts and uninterrupted prediction over the planned maneuver.
5. Establish command-driven drone/cable prediction before attributing deployment
   failures to cable parameters.
6. Freeze the adaptation protocol, then collect and evaluate real adaptation trials.

The intended contribution is a measured, data-efficient improvement of aerial
open-loop manipulation with explicit separation of initialization, physical
identification, and model discrepancy. Whether that is sufficiently novel and
effective remains an experimental question. The paper should narrow its claims to
the evidence obtained, including negative results or an identified limit on the
open-loop horizon.
