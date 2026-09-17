# Framework completeness review — 15 September 2026

## Scope

Reviewed the live AeroWhip `main.tex`, including the problem formulation, DDER assumptions, predictor, planning, correction, refinement, and framework figure sources. Following the user's clarification, this report excludes experimental results, data provenance, and proposed experimental changes. No manuscript or implementation changes were made.

Manuscript: [main.tex](<C:/Users/wts28/Lehigh University Dropbox/TonyLehigh Wu/Apps/Overleaf/AeroWhip-ICRA/main.tex>).
Reviewed SHA-256: `2580fc3cd4471e097eca8138999edb3cab39f7e9fd9488d1c32dfe8076950e2e`.
Line numbers below refer to that version.

This is a source and formulation review, not a fresh physical validation, numerical convergence study, compilation check, or complete literature audit. Existing historical test reports were not counted as tests run for this review. Several planner versions coexist; their differences must not be silently combined.

## Essential framework additions

### 1. Name the physical and neural optimizers and distinguish their derivatives

**Location:** lines 760–827. **Status:** confirmed omission.

The stages and robust loss are described, but nonlinear least squares, TRF, and Adam are unnamed. The full parameter-fitting path uses `scipy.optimize.least_squares`; the older explicit call names `method='trf'`. Full refinement supplies GPU-batched central-difference Jacobians in logarithmic parameter coordinates. Neural fitting uses automatic differentiation through rollouts and Adam. Therefore, describing the simulator as differentiable does not imply that every optimization stage uses automatic differentiation.

Suggested concise addition:

> We solve each bounded nonlinear least-squares parameter subproblem using the trust-region reflective (TRF) algorithm, with central-difference Jacobians of the rollout residuals. We train the acceleration-residual networks using Adam and automatic differentiation through the complete rollout.

Evidence: [parameter solver and Jacobian](../../experimental_data/whip_full_optim.py), [cable parameter Jacobian](../../experimental_data/whip_full_cable.py), [neural optimizer](../../experimental_data/whip_full_continuation.py), [explicit TRF call](../../experimental_data/flight_adaptation.py).

### 2. Define the complete parameter-fitting objective

**Location:** lines 769–798. **Status:** confirmed omission.

The neural objective is explicit, but the physical-parameter prior is only described as penalizing changes. Define the actual stage objective, using a new parameter symbol to avoid collision with the command-control vector:

\[
\min_{\eta_{\min}\leq\eta\leq\eta_{\max}}
\mathcal L_{\mathrm{data}}(\eta)
+\lambda_p\|\log(\eta/\eta_{\mathrm{start}})\|^2.
\]

Division and logarithm are componentwise. For vehicle delay, identify the separate candidate search and its penalty relative to the preceding delay. The implementation compares each candidate's optimized loss plus `nominal_prior*((delay-prior.delay_s)/0.04)^2`.

Calling this nonlinear least squares is consistent with the robust loss: for `u=e/sigma`, the code supplies the transformed vector `r=u*sqrt(2/(sqrt(1+||u||^2)+1))`; its squared norm is exactly `2*(sqrt(1+||u||^2)-1)`. Observation weights and parameter-prior residuals are appended accordingly. The implementation applies robustness to each three-dimensional observation norm, not separately to its coordinates. A brief explanation is sufficient; the residual transformation need not occupy a main-text equation.

Evidence: `FullTranslation.capture_nominal/evaluate`, `fit_nominal`, and `FullCableForward.evaluate` in the files above.

### 3. Specify the admissible-command constraints

**Location:** lines 178–183, 667–678, 731–736. **Status:** confirmed omission.

The problem statement promises prescribed motion limits in the planning section, but that section only says violations make candidates infeasible. Define the constrained quantities: command and predicted vehicle altitude, speed, specific force, tilt, positive vertical specific force, cable ground clearance, and spline jerk. Identify which apply to commands and which to predictions. Numerical validity is a rejection condition, not a physical limit.

State that the spline's derivative control coefficients bound jerk throughout its spans, while other feasibility checks are evaluated on the relevant sampled command/rollout grids. Recovery must also pass checks. Do not imply an exact continuous-time feasibility certificate for every state constraint.

Evidence: [spline jerk bounds](../../planning/position_spline.py), [command and rollout checks](../../planning/reference_correction.py), [correction subproblem](../../planning/local_reference_correction.py).

### 4. Resolve the planner-version mismatch before adding more prose

**Location:** lines 587–678. **Status:** confirmed differences between the manuscript and inspected implementation paths; intended framework version requires a decision.

The manuscript combines hover-centered initialization without a supplied motion demonstration, a first-tip-entry event, and proximity/retraction/extension rewards. The inspected code has two distinct relevant paths:

- `mppi_trajectory.py` with `preferred_fold_v1` uses retained command seeds and an archived cable-shape/tangent-history preference. Its encounter is latched at first contact by any tracked cable site, and the contact/cast scores include a tip-first factor. The manuscript omits the shape term and tip-first factor and defines a different event.
- `strike_mppi.py` supports from-scratch hover initialization. Its `targeted_fold_strike_v1` objective uses a different distance/speed expression and event-selection rule in `strike_objective.py`.

Thus, the manuscript's equations should be explicitly bound to one intended planner. If the preferred-shape version is intended, disclose its seed, shape-reference term, and encounter rule. If the newer from-scratch version is intended, use its actual objective and event rule. This finding does not justify changing experiments or selecting a different algorithm without discussion.

Evidence: [older objective](../../planning/whip_objective.py), [older search](../../planning/mppi_trajectory.py), [from-scratch search](../../planning/strike_mppi.py), [newer objective](../../planning/strike_objective.py).

### 5. Complete the MPPI search definition

**Location:** lines 588–590 and 673–678. **Status:** confirmed omission.

The exponential weighting formula is present. Missing details include the proposal covariance/smoothness construction, how multiple means start, temperature adaptation, sample budget, and stopping rule. `adaptive_weights` adjusts temperature independently per proposal to a target effective sample size, computed as `1/sum(w_b^2)` over feasible random samples. Incumbents and deterministic baselines are evaluated separately from these weight updates. Temperature is not simply an unexplained fixed hyperparameter.

A concise algorithm box plus a small settings table would make this reproducible. Select settings from the intended planner version, since initialization and stopping differ between paths.

### 6. Define fitting termination and checkpoint acceptance

**Location:** lines 794–835. **Status:** confirmed omission.

The manuscript explains selection among candidate model combinations, but not termination within a fitting stage or selection of a network checkpoint. Explain the practical-plateau rule, retention of the best checkpoint, numerical validity/gradient checks, and fallback when a candidate is invalid. Distinguish a budget/manual stop from convergence. The precise stopping constants can be placed in a compact settings table.

The current neural fitting paths use Adam learning rate `0.001`, weight decay `0.0001`, and gradient-norm clipping at `1.0`. Weight decay is an additional parameter regularizer beyond the two stated acceleration-output penalties and should be disclosed if used in the intended framework configuration.

Evidence: [continuation/checkpoint handling](../../experimental_data/whip_full_continuation.py), [fit orchestration](../../experimental_data/whip_full_fit.py), [stopping settings](../../experimental_data/whip_full_data.py).

## Important implementation clarifications

### 7. Explain how history becomes an initial state

**Location:** lines 404–406, 446–463, 556–563. **Status:** partly described, calculations missing.

The causal history durations and frozen bias/alignment are already explicit. Add that velocities are estimated from local polynomial fits, cable states are reconstructed and projected onto length/velocity constraints, and missing-history handling is fixed before optimization.

Clarify that `b_q` is effective pre-hover compensation and `Q` is effective tracking-to-nominal alignment, not independently measured mounting calibration. Their estimates depend on candidate response parameters and must be recomputed consistently during fitting, then frozen during each rollout.

Evidence: [hover initialization](../../simulator/drone_pose_response.py), [causal cable velocity](../../experimental_data/current_adaptation.py), [parameter-dependent rollout](../../experimental_data/whip_full_optim.py).

### 8. State the numerical integration scheme and resolution

**Location:** lines 526–532. **Status:** qualitative solver split present; reproducibility details missing.

The manuscript already states implicit bending damping and mass-weighted position/velocity projections. Add the vehicle's midpoint integration and distinguish command update rate, outer prediction step, and cable substeps. Include node count, constraint iterations/tolerance, and curvature-frame regularization in settings. The retained model inspected uses 30 Hz commands, a 150 Hz outer step, eight cable substeps, four projection iterations, and twelve cable nodes; these are source-specific settings, not universal framework requirements.

Evidence: [vehicle integration](../../simulator/research_pose.py), [coupled rollout](../../simulator/research_execution.py), [DDER solver](../../simulator/cable/dder.py).

### 9. Finish the command-correction solver description

**Location:** lines 731–736. **Status:** main method already named; secondary details missing.

Damped bounded Gauss–Newton, finite differences, and backtracking are already stated. Add a short statement about the constrained local subproblem, adaptive damping/trust radius, stopping, and retaining the previous feasible command when no update is accepted. The inspected local subproblem uses SLSQP; this is a secondary solver detail, less important than naming TRF and Adam.

The control-point coordinates are scaled/conditioned, and controls with no effect on the executed prefix are held fixed. Mention the fixed prefix and feasible fallback rather than reproducing the entire conditioning implementation.

Evidence: [local correction](../../planning/local_reference_correction.py), [correction orchestration](../../planning/correction_job.py).

### 10. Provide the settings that determine objective balance

**Location:** lines 534–544, 602–603, 619–663, 720–725, 786–823. **Status:** symbols and architecture present; values/scaling incomplete.

Prioritize loss scales, nominal prior weight, half-tip/half-marker weighting, correction weights, parameter bounds, delay candidates, neural input scaling, learning rate, and stopping constants. The full-fit default uses position/cable scale 0.02 m, orientation scale 0.05 rad, nominal prior weight 0.03, and residual-output magnitude/change weights 0.01. These scales are objective design choices, not calibrated measurement-noise standard deviations. Use the intended frozen configuration when writing the final values.

Vehicle input normalization is implemented for position error, velocity error, desired acceleration, vehicle velocity, and bias. Cable velocity features are also scaled. The output bound is implemented with tanh. Architecture alone does not completely specify these networks.

Evidence: [fit defaults](../../experimental_data/whip_full_data.py), [vehicle features](../../simulator/drone_pose_residual.py), [cable features](../../simulator/cable/residual.py).

## Already covered; avoid duplicating

- Reduced centerline-only rod assumptions, free-pivot attachment, inextensibility, bending energy, and corotational damping.
- One-way vehicle-to-cable coupling and effective loading interpretation.
- Tracking-frame attachment offset and orientation prediction.
- Network sizes, inputs, world-frame components, and acceleration bounds.
- Nominal planning initialization versus causal measurement-based replay initialization.
- Measured-boundary cable fitting versus command-driven combined prediction.
- Staged parameter/network updates and effective rather than uniquely identified physical parameters.
- Fixed original tip-reference trajectory and timing, no repeated initial search during correction, and no online cable feedback.
- Combined model-selection rule and the previous-predictor fallback.

## Minimal revision structure

1. Resolve the planning version, then correct only the affected planning definition.
2. Add one compact physical-fitting objective and the TRF/Adam/derivative sentence.
3. Define the feasible command set and add stopping/fallback statements.
4. Add a concise numerical-settings table and initialization explanation.

No additional experiment is required merely to fill these framework-description gaps. Any new claim of convergence, identifiability, stability, or physical performance would require separate evidence.
