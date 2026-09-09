# Nominal loaded-drone pose response

Implemented 8 September 2026, **standalone and unfitted**. The existing selected
model, PPO, GUI execution path and physical calibration remain unchanged.
No parameter/residual fitting, policy training or flight sending runs here.

Implementation: `simulator/drone_pose_response.py`. Phase-aware data adapter:
`experimental_data/drone_pose_response_data.py`. Reproducible numerical probe:
`tools/check_nominal_drone_pose.py`.

## State and interpretation

The state contains the timestamp, tracked-origin position p_O and velocity v_O,
tracked-frame-to-world rotation R_WT, angular velocity omega_T, a frozen effective
hover compensation b, and a constant attitude-reference alignment R_CT with
explicit provenance. It is an effective response of the installed loaded setup,
not a center-of-mass rigid-body or motor model.

P/V is predicted at the OptiTrack cf_7 top origin. A is the underside attachment.
The conventional frame C is constructed from a desired specific-force direction
and yaw. C is **not automatically identified with firmware/IMU body axes**.

Two explicit alignment modes are supported:

- `explicit_calibration`: caller supplies R_CT. Merely passing this matrix does
  not mean the software verified a hardware calibration.
- `prehover_effective_alignment`: R_CT is anchored using the already observed
  mean hover orientation and the nominal hover attitude command. This is an
  effective relative-orientation reference; it may absorb steady hover tilt
  or tracking-frame alignment. It is not a measurement of firmware mounting.

The latter is used only for the current unfitted legacy-data probes. The same
constant alignment is held throughout each prediction. No future orientation
is used to update it.

## Nominal equations

Use the actual held P/V/A/yaw commands evaluated at t minus an effective delay d.
The initial model shares horizontal parameters, with separate vertical values:

    a_O = Kp (p_d - p_O) + Kd (v_d - v_O) + Ga a_d + b
    p_dot = v_O
    v_dot = a_O

Kp has units 1/s², Kd 1/s, Ga is dimensionless, and b is m/s². These are effective
response parameters, not firmware parameter values. The code supplies no default
fitted gains. There is no additional translational acceleration-lag state in
this first implementation; use the effective delay and existing dynamic state
before introducing potentially interchangeable delay/lag parameters.

For the nominal attitude command:

    s = a_O + g e_z
    z_C = s / ||s||
    h = [cos(yaw), sin(yaw), 0]
    y_C = normalize(z_C cross h)
    x_C = y_C cross z_C
    R_d,WT = [x_C, y_C, z_C] R_CT

Orientation follows a critically damped second-order response:

    e_R = Log(R_WT^T R_d,WT)
    omega_dot_T = e_R / tau_R² - 2 omega_T / tau_R
    R_dot_WT = R_WT [omega_T]_cross

The second-order form preserves the measured initial angular velocity. The
damping ratio is fixed at one, leaving one attitude response parameter tau_R.
It is a response timescale, not an identified inner-loop frequency or a claim
to reproduce the full Mellinger controller.

Gravity appears in the desired attitude construction. It is **not added again
to a_O**, which is already kinematic acceleration. No virtual policy force or
explicit cable reaction is added to this loaded-drone response.

Translation is an empirical acceleration response, coupled to the nominal
orientation target but not projected through a physical thrust/motor plant.
This limitation is deliberate: p_O is not known to be COM, and the observations
do not establish an independently identified thrust plant. The model therefore
does not certify mechanical command feasibility; that still needs a separate
vehicle-limit contract before PPO/export use.

## Pre-hover initialization

Initialization requires a finite contiguous measured pose history and fresh,
constant position/yaw hover commands with zero V/A and body-rate feedforward.
The model starts at the last measured timestamp. It does not extrapolate the
initial state to CSV onset.

P/R are the final measured pose. V and omega use 11-sample endpoint estimates.
For a history of duration T:

    mean_v = (p_end - p_start) / T
    mean_a = (v_end - v_start) / T
    b = mean_a - Kp (p_d - mean_p) + Kd mean_v

Mean position uses timestamp-weighted trapezoidal integration. Endpoint
velocities are computed from observations within the completed pre-hover
history. This estimates a constant effective compensation that balances the
mean observed motion. It is not a direct observation of the firmware integral
state and is held fixed during the short prediction.

Long recovery/hold behavior and changing integral compensation are not validated
by this approximation. Future fitting must retain this initialization procedure
and its limitations explicitly instead of freely changing per-take bias to fit
the maneuver targets.

## Attachment output

Given the fixed r in tracked frame T:

    p_A = p_O + R r
    v_A = v_O + R (omega_T cross r)
    a_A = a_O + R [alpha_T cross r + omega_T cross (omega_T cross r)]

The output exposes O and A P/V/A separately, plus R, omega and alpha. There is
no NN for geometry. These predicted attachment outputs are ready for later
cable-boundary integration; the current PPO/execution path still uses its
preserved historical model until a new bundle is fitted and selected.

## Commands, timing and numerical integration

`CommandSchedule` holds B×C×11 command arrays with native event timestamps,
validity, per-row cache expiry and an explicit support end. It rejects missing
input rather than imputing hover or extending the last row indefinitely.
The data adapter uses the phase-aware native controller table unchanged and
never appends the historical unexecuted CSV tail.

The engine currently accepts zero body-rate feedforward, matching all provided
legacy commands and the current fixed-yaw route. Nonzero yaw/body-rate
feedforward is rejected pending an explicit interface/frame model; it is not
silently ignored. Absolute yaw commands remain supported.

Integration uses a second-order Lie midpoint method. It lands on each delayed
command event and requested output timestamp, and subdivides further according
to the maximum step. An exact 30 Hz command stream remains spaced at 1/30 s;
it is not rounded to the nearest 100 Hz tracking sample. A 5 ms maximum step
is used in the numerical probes; it is not an onboard control rate.

The step must be at most tau_R/4. Singular attitude commands and attitude errors
near 180 degrees are rejected as outside this local model's domain. These are
numerical/domain checks, not validated aircraft tilt limits. Output acceleration
uses the right-limit command at its timestamp and can jump at packet events.

The implementation review also added a translational resolution check:
`dt * max(Kd + sqrt(Kp)) <= 0.25`. This prevents a fast fitted translation
response from using a step selected only for slower attitude dynamics. It is
a conservative numerical resolution rule, not a measured vehicle limit or
an error/stability guarantee. Fitting still requires timestep-convergence checks.

The continuous dynamics are differentiable with respect to the gain/time-scale
tensors. Delay is an explicit schedule value; future identification should
search delay separately rather than pretending the discrete event selection
has a smooth delay gradient.

## Implementation review additions

The adapter now honors saved position/orientation validity masks both during
hover initialization and when returning diagnostic targets. Attachment validity
requires both pose components. It requires the requested hover duration, with
one observed sample interval of endpoint tolerance, and verifies the saved
FullState column order. Invalid query timestamps are rejected.

Maneuver-only replay (`post_hold_s=0`) keeps all recorded timestamps through
the observed CSV interval, even when the phase boundary falls between tracking
samples. Recording coverage is checked separately; no interpolated terminal
measurement or unexecuted command is inserted. Context records requested and
actual output end times and incomplete post-hold coverage. Diagnostic targets
also retain command validity, CSV row identity and phase-boundary uncertainty.

Legacy post-hold commands were chosen from the real measured end position.
Replaying those logged commands is a conditional response check, not an
autonomous prediction of a preplanned recovery. Score that phase separately.

The pre-hover compensation depends on nominal gains. A future optimizer must
recompute `initialize_from_hover` from the same fixed past observations for
each parameter evaluation; caching the compensation calculated under old
gains is inconsistent. The data adapter is an unfitted trial loader, not a
ready-made optimizer. CPU and CUDA gradient checks cover this dependence.

The nominal attitude target `a_O + g e_z` is an effective construction. With a
loaded cable and a tracked origin distinct from COM, it is not an identity for
actual motor-thrust direction. Assess orientation and attachment errors as well
as position error before accepting this low-parameter approximation.

See [the implementation review](NOMINAL_DRONE_IMPLEMENTATION_REVIEW_20260908.md)
for verified fixes, independent tests and remaining integration work.

## Verification and next fitting step

Latest implementation review: **59 targeted tests passed on Windows/RTX 4080**,
including the checks described above and adjacent geometry, legacy processing
and execution tests. All three unfitted recording probes remain numerically
identical to the original examples. See the linked implementation review for
the fixes and remaining modeling/integration limits.

29 targeted tests passed on Windows 11, including RTX 4080 CPU/GPU agreement.
Coverage includes analytic PD response and critically damped attitude solutions,
second-order timestep convergence, exact delayed 30 Hz pulse integration,
stationary hover with nonzero compensation, rotation-frame relabelling
invariance, attachment centripetal/tangential terms, gradient finite-difference
checks, missing-command rejection, measured timestamp enforcement and exclusion
of future measurements from initialization/prediction. Existing geometry and
legacy drone-residual tests also passed.

All three real whip takes completed finite unfitted probes on the RTX 4080:
`runs/audits/20260908-045331-240135-nominal-drone-pose/`. Probe gains are fixed
numerical examples (Kp=4, Kd=3, Ga=1, tau_R=0.08 s, d=0.02 s), not fitted or
selected model values. Recorded-vs-example plots are labelled accordingly.
No cable prediction, policy performance, recovery certification, Ubuntu test,
firmware deployment or real-flight validation was performed.

Next, define a nominal fitting protocol using complete observed maneuvers,
separate post-hold scores, causal initialization and explicit effective frame
alignment. Fit the nominal response before adding the drone residual; retain
parameter-confounding checks and the limited excitation of three similar whips.
