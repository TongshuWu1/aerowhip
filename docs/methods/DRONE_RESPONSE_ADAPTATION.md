# Learning drone response during sim → real → sim adaptation

Read the current [full-model adaptation contract](ADAPTATION_MODEL_CONTRACT.md).
Drone learning is one part of a workflow that also evaluates and adapts cable
physics and residuals. The single-gain branch below is a partial implementation.

The first real [gain-only adaptation](../development/M0_M1_FIRST_ADAPTATION.md) is now implemented
and completed under `whip_response_gain_v1`. It updates only nominal feedforward_xy
and is **not promoted** because held-out errors increased. Saturation, lag, broader
drone/NN fitting and physical capability identification below remain proposals.
The original diagnostic review below predates this narrowly scoped implementation.

10 September 2026. This is a literature/math review and tested diagnostic
implementation. It does not change the selected M0, MPPI commands, controller,
neural weights or any trajectory limit. No new real take has been fitted.

## Decision for our experiment

Learn the loaded drone **and its unchanged controller's command-to-motion
response** from logged PVA inputs and native measured pose. Feed that response
into the cable simulator and MPPI. A requested acceleration is not an achieved
acceleration. Exceeding the preliminary-data maximum is extrapolation, not an
automatic command rejection. The target is a better forward predictor, not a
hard cap chosen from a recording's largest acceleration.

The first task is to distinguish three quantities under the same actual inputs:
tracking error (measured minus commanded pose), model error (measured minus
predicted pose), and conditional cable error (using measured attachment motion).
The original saved M0 forecast remains the prospective prediction; reinitialized
diagnostics are separately labeled. A model can correctly predict imperfect
tracking. Changing it just to reduce commanded-versus-measured error would be wrong.

## What the primary literature supports

| Source and scope read | Relevant method | Transfer to our setup |
| --- | --- | --- |
| [Torrente et al., Data-Driven MPC for Quadrotors, Sections III.C–F and IV](https://rpg.ifi.uzh.ch/docs/RAL21_Torrente.pdf) | Augments nominal dynamics with learned acceleration residuals; forms training targets from next-velocity prediction error divided by timestep; evaluates tracking and new trajectories. | Correct a forward model and propagate it during planning. Their inputs are individual rotor thrusts and their residual features are body velocity; our PVA interface and cable-loaded response require different features/interpretation. Their actuator constraints are not a justification for imposing our training-data maximum on PVA commands. |
| [Eschmann, Albani and Loianno, Data-Driven System Identification of Quadrotors Subject to Motor Delays, Section III and IV](https://arxiv.org/html/2404.07837v1) | Identifies thrust/torque/inertia and a latent first-order motor response from motor setpoints and onboard inertial measurements, with a MAP formulation. | Delay is a real identification concern. Our PVA plus external pose logs can identify effective closed-loop response, but cannot be substituted for their motor/IMU observations to claim identified thrust curves. Their approximately minute-long, varied data collection is not evidence that one short whip identifies every parameter. |
| [Sun, de Visser and Chu, Quadrotor Gray-Box Model Identification from High-Speed Flight Data, author repository abstract](https://repository.tudelft.nl/record/uuid:afe95c6a-d2f1-4218-b97e-5e7cf54aa219) | Stepwise models combine physical knowledge with high-speed flight observations and compare prediction residuals. | Choose a small supported model extension and validate it. Only the repository abstract was available here; no detailed algorithm is attributed to it. |

These sources support the modeling approach. They do not establish our smoothing
windows, diagnostic scales, parameter probes or sufficient number of whip takes.
Those are explicit engineering choices to assess against the incoming data.

## Math and the current implementation

Write the observed input as
\(u(t)=[p_d,v_d,a_d,\psi_d]\), sampled with its actual zero-order-held clock.
The current nominal translational model at the tracked origin O is

\[
 a_0=K_p(p_d-p)+K_d(v_d-v)+K_{ff}a_d+b,\qquad
 \dot p=v,\quad \dot v=a_0+r_0(p,v,b,u).
\]

Commands are evaluated at \(t-d\), where \(d\) is effective command delay.
\(K_p\) has units s⁻², \(K_d\) s⁻¹, \(K_{ff}\) is dimensionless, and
\(b,r_0\) have units m/s². The CSV acceleration is kinematic; gravity must not
be added to it. The fixed nuisance \(b\) is estimated from past hold data, not
an observed firmware integrator state. The original residual \(r_0\) stays frozen.

Orientation has a separate second-order response driven by nominal acceleration:

\[
 \dot\omega\approx \tau_R^{-2}\operatorname{Log}(R^TR_d)^\vee
                     -2\tau_R^{-1}\omega.
\]

The code uses Lie-midpoint integration and exact delayed packet boundaries.
It does **not** derive translation from realized thrust along the predicted body
axis. Therefore changing \(\tau_R\) alone cannot correct translational lag in this
model. A bounded residual also does not bound the total acceleration, since
the nominal feedback/feedforward term can keep growing. Cable reaction is not
injected into drone dynamics; load effects are effective properties of the
fitted loaded system. Large state-dependent cable forces could violate that approximation.

For comparison, a physical COM equation is

\[
 m\dot v_C=m g_W+T R_{WB}e_3+f_{\mathrm{cable}}+f_{\mathrm{other}}.
\]

Identifying \(T\) or a motor limit requires separating the other forces and using
the correct body/COM frame. Our tracked origin is not automatically the COM or
the firmware frame. Rotating the measured attachment offset remains necessary
for the cable boundary. A world-X acceleration ceiling is not equivalent to a
thrust ceiling: available horizontal response depends on attitude and vertical
demand too.

## What a learnable response limit would mean

An illustrative one-axis smooth response is

\[
 a_{\mathrm{eff}}=A\tanh(gq/A),\quad A>0,\ g>0.
\]

Here \(q\) is an internal acceleration-like demand in m/s², not necessarily the
CSV acceleration alone. This equation is **not installed in the current model**.
It illustrates learning a diminishing response inside the dynamics without
clipping the generated PVA input. Applying it only to feedforward while leaving
feedback and residual terms unbounded would not produce a total-response limit.
A future extension must define the complete response, equilibrium, orientation
coupling and inherited residual consistently in all production/fitting solvers.

For \(z=gq/A\), the sensitivities are

\[
 \frac{\partial a}{\partial g}=q\operatorname{sech}^2z,\qquad
 \frac{\partial a}{\partial A}=\tanh z-z\operatorname{sech}^2z.
\]

When \(|z|\ll1\), \(a\approx gq-g^3q^3/(3A^2)\) and
\(\partial a/\partial A\approx 2z^3/3\). Low-demand data therefore give very
little information about \(A\), even with many repeated samples. Conversely,
deeply saturated data alone poorly constrain gain. Varied demand and time history
help distinguish gain, saturation and lag. A maximum observed acceleration is
neither a consistent estimate of \(A\) nor proof that it is a hardware limit.

A separate first-order lag, if justified, has the held-input update

\[
 x_{k+1}=e^{-\Delta t/\tau}x_k+(1-e^{-\Delta t/\tau})x_{\mathrm{demand},k}.
\]

This is different from a pure command time shift. Delay must be searched with
actual packet events, not gradients through silently interpolated timestamps.
Clock offset and effective delay can be confounded; neither is automatically
motor latency. No such extra lag state is added by this audit.

## Fitting and evaluation contract

Choose the smallest update supported by adaptation takes 001/002. A gain/delay
update should be tried before a more complex saturation family when diagnostics
support it. The intended primary loss is recursive pose prediction, e.g.

\[
 L(\theta)=\frac1N\sum_i\frac1{|V_i|}\sum_{k\in V_i}
 \left[\rho(\|\hat p_{ik}-p_{ik}\|/s_p)
 +\rho(\|\operatorname{Log}(R_{ik}^T\hat R_{ik})^\vee\|/s_R)\right]
 +\lambda\|D^{-1}(\theta-\theta_0)\|^2.
\]

Here \(V_i\) is a fixed reviewed observation mask; weights give equal take
influence. Scales and prior strength must be declared before fitting, not tuned
against validation. The diagnostic uses squared scaled pose error, not this
proposed robust regularized fitting objective. Its 2 cm / 0.05 rad scales are
not measured sensor standard deviations. Initialization uses only past samples;
no future measured pose is fed into a recursive prediction. Parameters and any
model-dependent nuisance initialization must be re-evaluated consistently.

Derivative-based response plots are supporting evidence, not fitting targets.
Local cubic fits use contiguous valid native pose, with explicit edge/gap masks
and windows truncated before reviewed contact. Their symmetric windows are
retrospective; they must never initialize a prospective forecast. The report
compares two window sizes to expose smoothing dependence. Orientation and
position are fit directly to avoid treating noisy differentiated acceleration
as independent precise measurements.

Inspect dimensionless local sensitivity columns for small response or strong
dependence. The diagnostic probes horizontal feedforward ±10%, delay ±10 ms
(nonnegative), and attitude time constant ±10%, one at a time, around M0.
It is a local finite-difference screen, not a parameter fit, a confidence
interval, or a proof of global identifiability. It excludes validation data.
If only a local response gain is supported, call it that; if saturation remains
ambiguous, retain that uncertainty rather than publishing a guessed ceiling.

Cable damping is fitted separately using measured rotated attachment. Then check
the full command-driven prediction with the candidate drone/cable combination.
Freeze selection before take 003 validation diagnostics. A later newly planned
M1 flight tests prospective improvement; fitting one M0 motion does not establish
generalization to a new whip.

## Implemented entry point and remaining boundary

After the original forecast comparison and reviewed preparation:

```powershell
.venv/Scripts/python.exe tools/adapt_whip.py diagnose-drone --job runs/adaptation/M1-whip-first --output runs/adaptation/M1-whip-first/drone_diagnostic
```

The command verifies prepared inputs/source identities, reads adaptation takes
only, replays seven pose variants per take through production CUDA graph kernels,
and writes per-take PNGs, raw diagnostic arrays, scaled sensitivity reports and
evidence hashes. It neither changes a model nor selects the best probe. Small
derivative regressions and three-column SVDs run on CPU; physics/NN recurrence
uses the selected device. This avoids an unnecessary cable rollout for every
drone probe. STOP is checked between takes/probes; no automatic restart occurs.

The existing `fit` stage still supports reviewed scalar cable damping only.
Drone candidate publication, changed-drone same-flight comparison and saturation
fitting require a reviewed extension chosen after these diagnostics. They are
not falsely represented as completed by this new command. No incoming file
automatically triggers fitting, model promotion or another planner run.

Verification: `runs/audits/drone-response-readiness-20260910`. Tests cover physical
derivative units, missing samples/timestamp gaps, absent/confounded excitation,
short invalid command intervals and validation exclusion. A full synthetic
Windows/RTX 4080 float64 pipeline exercises drone diagnosis and the existing
reviewed cable fit. Independent numerical checks verify the illustrative
saturation derivatives and held first-order update; these are not physical fits.
