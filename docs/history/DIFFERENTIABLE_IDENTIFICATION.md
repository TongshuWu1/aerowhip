# Physical identification and neural motion correction

## Current UI workflow

Data & Calibration now has three steps: Recordings, Fit setup, and Results.
**Fit geometry and cable drag** fits the lateral attachment offset while holding
measured height and lengths fixed, then searches cable drag using fit takes only.
EI and Cb stay at the selected input values; the neural residual is off.
Use **Use candidate as fit inputs** to start from a reviewed candidate instead of
the active baseline. Each job snapshots its inputs and enabled recordings.

Review 2-second and, when available, 5-second prediction errors and the measured
versus predicted tip traces. The comparison is against that job's starting model,
which may differ from today's active baseline. These are development validation
results under measured attachment motion, not open-loop flight validation.
**Apply selected candidate** versions the complete model, including geometry,
drag, material values and solver settings. New PPO/SAC runs use it; existing runs
retain their saved models. Export includes PDF/SVG/PNG, source metrics and saved
trajectory arrays.

See [the calibration audit](CONSTRAINED_FIT_20260905.md) for the evidence behind
this choice. A half-second real-data gradient check failed against finite
differences; 50/100 ms checks passed on two states. Long-horizon Adam and the
neural residual are therefore excluded from the standard UI fit.

## Legacy experimental workflow

The following describes the older `differentiable_physics_residual` backend,
retained for reproducing saved experiments. It is not the current UI action.
It runs two
stages on the enabled preliminary recordings. Raw recordings and the protected
test take remain unchanged. Configuration is snapshotted into the fitting job.

1. A broad EI/Cb grid initializes a differentiable physical fit. Adam then
   updates positive, bounded log parameters using recursive motion predictions.
   Training progresses through 0.1, 0.25, 0.5, and 1-second horizons; selection
   always evaluates complete one-second windows. Each minibatch weights the training takes equally and cycles
   through shuffled windows. No measured interior state is injected after the
   initial condition.
2. The selected physical parameters are frozen. A small neural network learns
   bounded acceleration corrections for cable nodes. The root receives no
   learned acceleration. The correction enters before the implicit damping and
   length constraints at each physics substep, both during fitting and runtime.

The network is a motion-discrepancy model, not a change in the meaning of EI or
Cb. Its inputs are root-relative cable positions, root-relative velocities,
and root velocity, all from the predicted state. Its output is limited to
2 m/s² per coordinate by default. It may represent unmodeled external effects;
it is not constrained to conserve momentum. It is translation invariant but
does not impose rotation equivariance. It has two 48-unit tanh hidden layers
and a zero-initialized output layer. It requires validation on the intended
motion range before use outside the preliminary dataset.

New fits use measured initial positions and a past-only quadratic velocity estimate
over 11 samples (100 ms at 100 Hz), followed by position/velocity constraint
projection. The grid initialization and differentiable refinement use the same
initializer. Existing job snapshots without this setting retain the centered
offline estimator for reproducibility. A warm-start neural refinement rejects a
change in initialization. The centered derivatives saved in force datasets remain
available for offline force analysis; they are not used to initialize new fits.

The 2026-09-05 initialization benchmark compared this estimator with the centered
reference and a bounded DER history state fit. The history fit did not improve
forward prediction in that test and is not enabled in the fitting workflow.

Both stages use full backpropagation through the prediction window with
checkpointed blocks to control memory. Checkpointing does not detach the state
or truncate the gradient. A robust marker-position loss reduces sensitivity to
tracking outliers. The residual additionally has a small acceleration penalty
on initial states. Gradient clipping limits updates; finite-loss and
finite-gradient checks stop invalid optimization. The recorded gradient norms
are before clipping and should be inspected for poor conditioning.

Selection uses the lowest full **training** objective, including the initial
candidate, so an unsuccessful optimization cannot replace it with a worse
training candidate. Validation curves are recorded but do not select weights.
An update is one balanced minibatch, not one complete dataset epoch. Defaults
allow 60 physical updates and 120 residual updates, with earlier stopping after
six full training evaluations without a meaningful improvement. Reaching an
update cap is not evidence of convergence.

## Evaluation and application

Each completed fit compares active physics, fitted physics, and fitted physics
plus NN on identical validation windows at 1, 2, 5, and 7 seconds. Different
horizons can have different available windows; the horizon plot is aggregate
window error, not one trajectory's error versus elapsed time. Per-take errors
and errors at individual lead times are retained in `fit_result.json`.

The first validation trace includes measured, active, physical, and hybrid
marker positions. Figures are exported as PDF, SVG, and PNG with CSV source
data. These are single-fit diagnostics; they are not confidence intervals or
multi-seed results.

The UI applies fitted physics by default. A separate checkbox includes the NN
only when it improves marker RMSE over fitted physics at every evaluated
horizon, with coverage of all configured horizons. That check establishes a limited validation result, not guaranteed
accuracy. Applying creates a versioned baseline containing immutable neural
weights and their SHA-256 hash. The runtime verifies that hash and uses the
same correction inside the DDER transition, including PPO/SAC simulation and
open-loop planning. Applying a new physical-only fit or manual baseline removes
an older correction so it cannot silently accompany changed physical inputs.

Prescribed-root fitting validates cable response to known attachment motion.
It does not establish accuracy of the force-controlled drone trajectory or
full sim-to-real strike performance. Neither PPO nor SAC is automatically
retrained by this operation.

## Relationship to DDER/DEFORM

This adopts differentiable recursive trajectory fitting and a separate learned
correction, but is not an exact reproduction of DEFORM. Our one-point pivot,
Kelvin–Voigt damping, robust loss, bounded MLP correction, and Adam optimizer
differ from the paper's setup. The grid remains an initialization and reference.
The physical solver uses float64 direct Cholesky damping and an equivalent
dense constraint solve for fitting. A closed-form isotropic bending gradient
is checked against the existing energy-autograd implementation, including its
derivatives. Runtime defaults retain their existing force and constraint paths.

Reference: [Chen et al., Differentiable Discrete Elastic Rods for Real-Time
Modeling of Deformable Linear Objects](https://arxiv.org/html/2406.05931v3).
