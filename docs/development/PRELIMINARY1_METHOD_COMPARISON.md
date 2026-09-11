# Preliminary1 fitting: comparison with published methods

Reviewed 9 September 2026. This is a method review, not a new fit or a claim of
flight validation. The user paused further implementation to examine whether our
approach is unnecessarily complicated or incorrectly implemented. MPPI remains
stopped; published preliminary M0 and all existing evidence are preserved.

## Assessment

Physics plus a learned correction is a defensible architecture. What our current
evidence does not justify is treating the completed optimization as a sufficiently
identified model for precise whipping. Initial-state sensitivity, weak physical
parameter movement, and unreliable long-window derivative checks precede any
question about increasing MPPI samples or residual training time.

The most useful simplification is to establish a trustworthy physical baseline
and its initialization before adding flexibility to the fitting problem. This
does not mean replacing a bending cable with a single pendulum: the latter would
remove the shape degrees of freedom needed for a travelling bend.

## What comparable work actually does

### Aerial cable identification and control

Shen, Franchi and Gabellieri's aerial-cable study uses a Crazyflie-based quadrotor,
OptiTrack, and a one-ended cable. For identification, the drone is fixed and the
displaced cable is released. They measure geometry/density and fit two material/
drag coefficients to seven seconds of marker motion, initializing from a static
equilibrium. A reduced model then supports feedback control. Their reported
100 Hz controller uses only a 30 ms prediction horizon. These are shape-control
experiments, not evidence that a 30 ms horizon can discover our whip. The release
comparison also uses identification data, so it is not an independent test.
[Paper, sections VI-A/B](https://arxiv.org/html/2403.17565v2).

### Initial state is an identification problem

Mamedov and colleagues' single-trajectory study uses a chain model with learned
joint forces. Training optimizes initial states alongside model parameters;
testing estimates state from preceding measurements using moving-horizon
estimation. Their preprocessing calibrates coordinate transforms, resamples data,
and applies zero-phase filtering before numerical differentiation. Such filtering
is retrospective, even if the subsequent estimator uses preceding samples.
Their experiments involve an aluminum rod and foam cylinder, not our free cable.
The lesson is to separate offline state reconstruction from deployable causal
initialization. Their particular 3.5 Hz filter should not be copied into whipping
data without checking the motion bandwidth.
[Methods 4.3 and appendix A, 2024 manuscript](https://arxiv.org/html/2407.03476v1).

### Differentiable rods and residual learning

DEFORM combines differentiable elastic rods, a neural integration correction,
and explicit inextensibility handling. It trains on one-second recursive
predictions and evaluates longer rollouts; its ablations favor multi-step over
single-step training. It uses 350 seconds of 100 Hz motion-capture data per
object. Its main setup controls both ends and supplies initial position and
velocity, unlike our free-tip aerial problem. This supports our broad model
family, but does not validate our solver, residual architecture, or initializer.
Short-window fitting may help diagnose our pipeline; short-window accuracy alone
cannot qualify it for a full maneuver.
[DEFORM, sections 3, 5.1 and appendix A/B](https://arxiv.org/html/2406.05931v2).

### Gradient-free physical fitting is a legitimate baseline

Lim and colleagues' Planar Robot Casting tunes simulator parameters using
Differential Evolution and then learns actions from physical and simulated data.
This demonstrates a practical alternative to differentiating through every
simulation step. The task involves a cable sliding on a surface, so its friction
model and achieved errors are not directly comparable to an aerial whip.
[Paper](https://arxiv.org/abs/2111.04814).

Yang, Stork and Stoyanov likewise compare parameter optimizers for DLO shape
control and report differential evolution as effective before generating
simulation training data. This observation is supported by the publisher's
abstract; detailed preprocessing was not verified here.
[Publisher article](https://doi.org/10.1016/j.robot.2022.104258).

### Whipping can use a small action description

Edraki and colleagues' 2025 workshop paper combines preparatory and striking
minimum-jerk motions, described by nine parameters. Its optimization penalizes
tip-to-target distance and effort, and it reports a physical Franka demonstration
with a 3D-printed whip. This is a limited demonstration, not a broad aerial
benchmark. It suggests a useful planning baseline: optimize a few smooth motion
parameters before searching a long sequence of independent control actions.
It does not establish our specific bend-stage reward as necessary.
[Workshop paper](https://deformable-workshop.github.io/icra2025/spotlight/01_01_05_Edraki_Human.pdf).

### Recent rope-specific identification

The April 2026 Wiggle and Go preprint uses an observation motion to infer rope
parameters, with a direct CMA-ES parameter-fitting baseline. It checks a different
motion as well as downstream tasks. Its fast learned inference depends on
substantial prior simulation training: 9,000 training ropes. This is useful
evidence for excitation and cross-motion validation, rather than a reason to
build another large network for our five takes.
[Preprint, methods and transferability evaluation](https://arxiv.org/html/2604.22102v1).

## Comparison with our measured evidence

| Part | What is defensible now | What needs correction or qualification |
| --- | --- | --- |
| Raw data | Preserved raw global XYZ, native markers, missingness, source hashes and whole-take split | Cached controller XYZ is the same OptiTrack source, not an independent pose sensor |
| Cable input | Fit cable with measured, rotated attachment motion | This is conditional cable prediction, not command-to-flight validation |
| Initial state | A causal history is appropriate for deployment | Earlier windows used inconsistent durations; one uniform second did not consistently improve prediction |
| Physical fit | Fit a small set of positive physical coefficients | EI/Cb barely moved after the coarse search and sit near its extremes; identifiability is unestablished |
| Residual | A bounded correction can address missing dynamics | Approximately 0.44% fitting-objective improvement and extreme gradients do not justify trusting it |
| Validation | Separate command-driven coupled rollouts exist | Representative held-out two-second tip RMS is 64.6 cm; a completed job is not a successful precision model |
| MPPI | Genuine search under the new M0 is preserved | More search cannot establish physical accuracy or repair a biased dynamics model |

Local evidence: [fitting audit](PRELIMINARY1_FIT_AUDIT.md),
[completed fit](PRELIMINARY1_FIT.md), and
[initial research design](../methods/IDENTIFICATION_RESEARCH_DESIGN.md).
The earlier research design already calls for separating state, physical
parameters and residuals. Our implementation and acceptance criteria have not
yet fulfilled that design.

### Checks completed before the literature pause

The earlier user-authorized work stopped MPPI run `20260909-224132-314704` at
40 committed commands / 1.3333 simulated seconds. No new model was fitted.

An optional exponentially weighted quadratic endpoint-velocity estimator was
implemented using the requested full second of past samples. A training-only
50 ms prediction score favored a 0.02 s weighting time constant: position RMSE
3.39 cm with uniform weighting versus 0.44 cm with this weighting. This places
most effective weight on very recent samples despite retaining one second of
history. It is not evidence of better full-maneuver predictions, nor a selected
new M0. Old protocols retain their previous behavior.
Evidence: `runs/audits/preliminary1-refit-checks/estimator_selection.json`.

The gradient diagnostic used two training windows and perturbed one residual
output-bias coordinate, comparing recursive autograd with central differences.
At 40 ms they agree closely. At 0.2, 0.5 and 1 s they do not agree consistently
across perturbation sizes. At 1 s, autograd gives about 40.39 while numerical
differences range from 0.84 to 21.09. This fails the intended fitting check; it
does not isolate an autograd implementation bug from numerical ill-conditioning
or nonsmooth dynamics. A reference implementation comparison remains unfinished.
Evidence: `runs/audits/preliminary1-refit-checks/gradients.json` and its script.
These checks ran on Windows / RTX 4080; no other platform was verified.

## Recommended next procedure — proposed, not started

1. Keep the existing recordings and frozen M0. Retain the raw/missingness/clock/
   geometry checks; these are necessary preparation, not disposable complexity.
2. Establish a cable-only reference using measured attachment motion and zero
   learned cable correction. Reconstruct positions and velocities offline with
   explicit noise/length constraints, alongside the separate causal initializer.
   Compare their prediction errors so initial-state error cannot silently become
   damping. Any state adjusted using a scored future segment must be labeled
   retrospective, not used for a deployment-accuracy claim.
3. With geometry and masses fixed, map the EI/Cb loss surface and use a bounded
   derivative-free reference search. Batch candidates/windows on the GPU. Check
   timestep and forward-simulation stability first: derivative-free search avoids
   unreliable gradients, but cannot cure an inaccurate forward simulator.
4. Assess uninterrupted 0.25, 0.5, 1 and 2 s forecasts across motion families.
   Select using training/development data only. `figure8_002` stays excluded from
   fitting/selection, but it has now been repeatedly inspected and should not be
   called a fresh blind test. A later prospective take supplies new evidence.
5. If the few-parameter model cannot explain the data, examine attachment behavior,
   effective damping, marker mass distribution and excitation before adding a
   neural correction. A separate fixed-attachment release take would help isolate
   cable behavior, but is a suggested new experiment, not currently collected data.
6. Validate the command-to-drone block separately, then combine it with the cable.
   Add a small residual only if its improvement survives the same independent
   rollout checks. Resolve long-window gradient behavior before residual training.
7. Once prediction is adequate, compare existing MPPI with a small smooth
   preparation/reversal motion family, allowing height to be optimized. Evaluate
   target distance and actual wave propagation separately. Changing the optimizer,
   reward or action parameterization would be a new experiment, with preserved
   original settings and forecasts.

These recommendations are our inference from the literature and local evidence,
not a single published recipe or a guarantee that two cable coefficients suffice.
They aim to reduce ambiguity and unnecessary optimization, while retaining the
degrees of freedom needed for the user's horizontal travelling-wave strike.
