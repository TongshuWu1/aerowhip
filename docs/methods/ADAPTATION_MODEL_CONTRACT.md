# Project and full-model adaptation: current contract

Latest method design: [systematic adaptation protocol](SYSTEMATIC_ADAPTATION_PROTOCOL.md).
The user classifies current recordings/runs as development and will collect a
separate clean study after refinement. The proposed combined fitting objective
extends the staged workflow described below; it has not been implemented or run.
M1-full has flown, and staged M2 has completed with mixed complete prediction.
Use HANDOFF for current evidence and the linked protocol for future study design.

Reviewed 10 September 2026 against the original research proposal, selected M0
assets, production predictors, preliminary fitter and current whip-adaptation CLI.
This restores the intended scope after the user corrected the interpretation of
the first M1 trial. It is a design/code audit, not a newly fitted model.

Subsequent explicit authorization implemented the complete workflow. Its
`runs/adaptation/M1-full-whip-v2` continuation is completed and registered without
promotion; the failed v1 and earlier gain-only M1 remain preserved. Read
[the implementation and measured-run report](FULL_MODEL_ADAPTATION.md) and the
[window, residual and limits review](../development/TRAINING_HORIZON_RESIDUAL_REVIEW.md).
Missing-implementation statements below record the audit finding
that motivated that work; they no longer describe the available full fitter.

## Research objective

Improve a reusable drone–cable forward model between real flights, then use it to
plan a better aerial whip. The desired behavior is a forward preparation and
backward release that propagates the cable toward a target beyond the drone.
MPPI is the current planner; PPO is a separate way of obtaining a maneuver.
Changing a reward or reproducing one recorded trajectory is not model adaptation.

The evidence chain is:

**Preliminary recordings → M0 → planned commands and frozen forecast → measured
flight → model adaptation → candidate M1 → same-input validation → new plan and
frozen forecast → next measured flight.**

Updates happen between flights. The PVA command sequence is fixed during the
whip, while the onboard controller continues tracking it. Each model generation
is a complete package of nominal parameters, residual weights, geometry,
initialization conventions and data/source provenance. The generation number
does not imply improved accuracy or physical success.

The original [research proposal](../paper/RESEARCH_PROPOSAL_ADAPTIVE_AERIAL_WHIP.md),
especially hypotheses H1/H2 and Section 5, already calls for aircraft/cable
identification and a restrained residual with preliminary-data retention. Its
old force interface, vehicle identity and timing are historical; the current
interface is recorded tracked-origin PVA at 30 Hz for the 145 g drone and 17 g
cable assembly.

## What is in the model

There are two dynamical subsystems, each with a possible learned correction:

| Block | Input and prediction | Current selected M0 |
|---|---|---|
| Nominal drone response | Recorded delayed PVA/heading and initial pose → tracked-origin position, velocity and orientation | Effective loaded-drone feedback/feedforward and attitude response |
| Drone residual | Predicted motion, P/V errors, commanded acceleration and past-hold compensation → acceleration correction | Enabled inherited network, two hidden layers of width 16, bounded per-axis correction ±0.5 m/s² |
| Nominal cable physics | Attachment motion and initial cable state → node motion | DDER, measured geometry/masses, inherited EI/Cb, effective damping 0.4/s |
| Cable residual | Relative cable geometry/velocities and attachment velocity → free-node acceleration correction | Disabled in the selected M0 development variant |

The residual is the remaining **model prediction error**, not the distance from
the tip to the target and not simply desired minus measured drone position.
An accurate model should predict imperfect tracking when that is what occurs.

The current drone translational equations are

\[
\dot p=v,\qquad
\dot v=K_p(p_d-p)+K_d(v_d-v)+K_{ff}a_d+b+r_D(p,v,b,u).
\]

Here the recorded command is evaluated at the effective delay, and b is estimated
from past hold observations and frozen over the short prediction. It is not the
measured firmware integral state. The residual acts on acceleration; attitude
has a separate nominal response driven by nominal acceleration. Attachment
position is **A = p + R r**, using the saved tracking-frame offset. Cable physics
and any cable residual then advance under this boundary, with rod constraints.

The present execution model is a cascade: predicted drone → predicted attachment
→ predicted cable. It does not return computed cable tension to the drone.
Loading is absorbed into the empirical loaded-drone model. Its residual has no
cable-state input, so it cannot explicitly distinguish different cable loads at
the same drone state and command. This is a limitation to test, not a reason to
add a second cable force without reformulating the existing loaded response.

Sources: `simulator/drone_pose_response.py`, `research_pose.py`,
`drone_pose_residual.py`, `research_execution.py`, `cable/residual.py`.
The cable residual implementation is a fixed-node full-state MLP, not the shared
local network proposed in older research notes. It excludes target, take ID and
trajectory time. Its corrections are effective discrepancies, not identified
elastic forces or a guarantee of momentum conservation.

## How the complete adaptation should work

**1. Prepare one reviewed dataset.** Keep raw global observations and missingness;
verify commanded packet identity and timing. Keep whole-take roles. Use causal
one-second cable and 0.4-second drone histories. Constrain nuisance initialization
and clock uncertainty independently of the fitting target; never use future
flight observations to improve a claimed prospective initial state.

**2. Update the nominal drone response.** Start from M0. Use actual logged PVA
inputs and measured native drone pose, with recursive predictions. Assess
horizontal and vertical response, timing and attitude separately before choosing
an identifiable parameter subset. Keep the inherited residual fixed during
nominal fitting so both parameter sets do not compensate each other freely.
The scope must not default to one horizontal gain. Translation lag and attitude
lag are distinct in this implementation. Saturation/capability requires a
supported model family and informative data; an observed acceleration maximum
is not a learned physical ceiling.

**3. Update the drone residual.** Freeze the chosen nominal drone parameters and
fine-tune the inherited residual on the remaining repeatable prediction error.
Use measured position trajectories as the primary supervision through recursive
rollouts. Differentiated pose can diagnose acceleration error but is noisy and
is not an independent precise acceleration sensor. Penalize large corrections
and changes from the inherited response. An attitude error cannot be assigned
directly to this acceleration-only network.

**4. Update nominal cable physics.** Drive the cable with the **measured rotated
attachment trajectory**, not the imperfect predicted drone boundary. Fit cable
observations across the maneuver, including the tip. Assess EI, bending damping
and effective external damping for sensitivity and ambiguity; do not force all
coefficients to change or claim precise material identification when they trade
off. Measured geometry and masses remain fixed. A low conditional cable error
helps locate error but does not prove cable parameters need no further review.

**5. Train the cable residual.** Freeze the chosen cable physics and learn its
remaining repeatable motion discrepancy under the same measured boundary. Since
the selected M0 has no cable residual, a candidate must explicitly introduce a
zero-output network and its specification; no historical cable weights should
be restored. A bounded acceleration correction can coexist with fixed 0.4/s
damping. The current loader rejects learned-damping modes with nonzero separate
damping: choosing those modes requires an explicit model migration, preserving
the baseline response rather than silently removing or double-counting damping.

**6. Assemble and validate the whole candidate.** Run recorded PVA through the
updated drone plus cable models, initialized once, with no measured future
boundary. Report drone XYZ/attitude, attachment, marker and tip errors; distinguish
early and late behavior and individual takes. Retain the intermediate nominal
and residual stages so their contributions are visible. Reject an update that
only improves an average by concealing important component regressions.

These are stages of one adaptation workflow, not mutually exclusive choices
between adapting the drone and adapting the cable. Every block receives an
explicit evaluated/updated/retained decision. Full adaptation does not require
every parameter or neural weight to change regardless of evidence.

## Objective, data reuse and efficiency

For a block B, use an explicitly specified loss of the form

\[
L_B = L_{\mathrm{whip},B}
      +\lambda_{\mathrm{replay}} L_{\mathrm{preliminary},B}
      +\lambda_\theta\|D^{-1}(\theta-\theta_0)\|^2
      +\lambda_r\mathbb E\|r\|^2
      +\lambda_\Delta\mathbb E\|r-r_0\|^2.
\]

Each trajectory loss averages fixed valid observations with deliberate equal-take
weighting. Position/orientation scales and marker/tip weights must be declared;
they are not sensor standard deviations without calibration. The equation is a
proposed staged objective, not the currently implemented single-gain loss.

Reuse a declared subset of **preliminary training data from the same repaired
system** to retain broader motion coverage. Never mix in validation recordings
or archived different-system data. Whip recordings teach the task-relevant
regime; varied preliminary motions help distinguish response parameters.
Splitting one flight into many overlapping windows does not create independent
experiments. Short training windows are acceptable, but the decisive comparison
is a complete free prediction over the maneuver.

Reuse production float64 CUDA equations, batch independent windows/candidates,
and retain the verified solver and full temporal gradients. Check representative
whip gradients before neural training, since earlier cable conditioning issues
were configuration dependent. Practical plateau stopping retains best/current
weights, optimizer and provenance. Maximum GPU utilization is not itself a
scientific or runtime objective; avoid redundant replays and CPU transfers.

## What is implemented, and what is missing

| Capability | Actual status |
|---|---|
| Nominal drone, drone residual, nominal cable and cable residual fitting | Present in the preliminary fitting code, with its own data/window contract |
| Immutable raw whip preparation, original-forecast comparison, measured-boundary and command-driven diagnostics | Implemented |
| Whip nominal cable update | Current `fit` entry supports scalar external damping only |
| Whip nominal drone update | Current `fit-response` entry supports feedforward_xy only |
| Both residual training stages in the raw whip workflow | Not wired into `adapt_whip.py` |
| One combined drone/cable/residual candidate and comparison contract | Needs extension beyond the current scalar scope checks |
| Actual M1 trial | One changed drone gain; cable and both residual states unchanged; not promoted |

Do not launch the preliminary fitter on whip inputs as if these contracts were
interchangeable. The next implementation task is to connect these stages to the
reviewed raw-whip data and combined candidate contract, retaining provenance and
production parity. No new fitting or model promotion follows from this document.

## Why the first M1 does not establish failure of the research idea

The original candidate reduced horizontal feedforward 0.74668→0.61834. Training
drone RMS improved, while take 005 drone/tip RMS worsened 8.93→10.41 cm and
14.89→17.93 cm. The audit in `runs/audits/M1-regression-audit-20260910` found:

- Exact reconstruction of prepared time/pose/sites/packets from raw inputs on
  all four included takes; histories remain causal and original hashes intact.
- M1 attitude RMS worsened on every included take. Training take 002's actual
  robust pose loss also worsened, despite the mean training objective improving.
- Both models predict late Z approximately 12–15 cm too low. The single changed
  horizontal gain cannot address that common error. Measured peak X ranges
  approximately 0.630–0.853 m across the included repeated flights.
- On 005, fixed M1 remains worse in both declared ±20 ms clock scenarios.
- Independent reference versus production position difference is below 0.007 mm;
  halving the reference step changes position below 0.015 mm on checked take 001.
  This does not explain centimetre-scale regression or prove all model equations
  are physically sufficient.

These identify an inadequate update scope and mixed component outcomes. They do
not identify a particular hardware fault, prove a saturation mechanism, or
establish that initial-state uncertainty is absent.

001/002/004 remain adaptation. 003/005 retain their reserved roles; 003 cannot
pass the current strict cable initializer. 005 was untouched by the first fit,
but has now been inspected during method review. Any redesigned method evaluated
again on 005 must disclose that reuse; a new frozen-plan flight supplies the next
prospective evidence. Do not fit 005 or relabel old predictions.

## Connection to primary literature

[SimOpt, Section III](https://arxiv.org/pdf/1810.05687) alternates real experience,
simulator-distribution updates and policy optimization, with a trust-region
restriction on distribution changes. It supports the overall loop and explicit
uncertainty; it does not guarantee each newly named model improves every flight.

[Torrente et al., Sections III–IV](https://rpg.ifi.uzh.ch/docs/RAL21_Torrente.pdf)
learn acceleration discrepancies over nominal quadrotor dynamics and test unseen
trajectories. Their rotor-thrust interface differs from our closed-loop PVA
interface. Our residual features and identification claims must match our data.

[DEFORM, Sections 4.2 and 5, Appendix A.3](https://arxiv.org/html/2406.05931v2)
combines rod parameter learning, residual corrections and multi-step training.
This supports assessing physics and residual contributions together. Its
integration corrections and two-manipulator boundary differ from our free-tip
aerial cable; our code is not a literal reproduction of that method.

[Mamedov et al., Sections 4.3–5.1](https://arxiv.org/html/2407.03476v1) distinguish
model parameters from initial-state estimation and evaluate different motions.
This supports separating causal initialization from dynamics and testing reuse
beyond repeated execution of one command.

Our proposed staged ordering, replay weighting and acceptance checks are project
design choices informed by these sources, not a claim to reproduce one paper.
