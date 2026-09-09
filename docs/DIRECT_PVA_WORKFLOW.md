# Direct PVA aerial-whip workflow

9 September 2026. Active branch: `twin-rewrite`. User authorized a fresh normalized-adp0 bootstrap, direct PVA PPO/MPPI and a UI overhaul. PhysX work remains paused in the separate Simulator worktree.

Latest scope: fitting and PPO are stopped while the user investigates drone
tracking. MPPI verification with frozen historical normalized M1 is complete;
no further diagnostics restart automatically. The fresh M0 is unfinished and
has not been published. Read [HANDOFF.md](../HANDOFF.md) for current status.

Latest task correction: MPPI must perform a forward pull followed by backward
drone release before the tip hits. A forward-only fast contact is insufficient.
Read docs/MPPI_PULLBACK_20260909.md for the explicit ordered motion thresholds,
new phase rewards and verified2s simulation. Historical tip-only results are documented in the archive; they do not
validate the current forward-pull/backward-release task.

## Commands and model

Both planners choose three bounded XYZ jerk values per 1/30 s. With interval h:

```
p_next = p + h v + h² a / 2 + h³ j / 6
v_next = v + h a + h² j / 2
a_next = a + h j
```

These are tracked-origin world coordinates in metres, m/s, m/s² and m/s³.
Acceleration in the CSV is kinematic; do not add or subtract drone/cable weight.
Yaw is fixed, body-rate feedforward zero. Packets are held at 30 Hz. The first
packet is the initial hover P/V/A; a jerk choice changes the next packet. This
causality and the fitted command delay are included during training.

The exact held packets drive the existing fitted loaded-drone pose response
and its bounded NN. Predicted orientation rotates the tracking-to-attachment
offset. That predicted attachment drives the DDER cable and its NN. The cable
physics advances at 150 Hz, with its internal solver substeps. No virtual point
force, force-to-reference conversion or measured future cable state enters new
PVA planning. Export the desired input P/V/A, not the predicted actual motion.

The drone is an effective closed-loop loaded response, not an identified motor
or battery model. Explicit cable reaction is not added again to this model.
The cable remains a point-attached, freely pivoting rod, not a suspended point
payload. Its NN includes nonnegative damping and a bounded acceleration
correction. Separate fixed drag is zero. New cold drone residuals have a smooth
zero-at-rest gate, leaving the nominal hover compensation responsible for the
static equilibrium. Historical residuals retain their original equations.

## Two independent planners

- PPO uses simulated state and pending command history to generate the offline
  jerk sequence. Initial input is the desired settled drone origin and target;
  cable hanging/zero velocity is assumed. Its internal simulated cable state
  does not require a real-time cable measurement during flight.
- MPPI now uses a **rolling 2-second lookahead**, independent of total
  maneuver duration. At each simulated state it samples 1024 temporally
  correlated Gaussian latent sequences plus the deterministic mean, maps them
  through tanh to bounded jerk, and evaluates 60 actions. It commits only the
  first selected action, shifts the proposal and replans from the continuing
  state. Pose, cable state, delayed command queue, contact history and absolute
  time are preserved. The final saved plan contains the committed actions.
  Candidate hits are predictions; only the independent committed simulation
  determines reported success. This is offline planning, not aircraft feedback.
  A batched screen of321 bounded forward/backward jerk guesses initializes the
  first proposal; MPPI then freely refines it. Ordered forward pull/backward
  release is required using modeled drone motion at interpolated tip contact.
- The proposal uses temperature-scaled importance weights. An explicit
  `control_prior` coefficient controls the Gaussian change-of-measure penalty;
  its current zero setting removes the previous additional pull toward zero
  jerk. Geometric terminal guidance is disabled after the longer-horizon
  comparison. Explicit strike-exit costs favor low speed, low upward speed
  and low positive acceleration along velocity at handover to recovery.
  These costs are separate from task return and do not alter physical hit rules.
  No CEM elite update, maneuver spline, PPO checkpoint or training is involved.
  Historical mode-less/open-loop jobs retain their original whole-whip semantics.

Settings live independently in `config/pva/ppo.json` and `mppi.json`; runs in
`runs/ppo_pva` and `runs/mppi_pva`. Every run owns model assets, settings and a
source snapshot. New PPO checkpoints use `jerk_pva_ppo_checkpoint_v1`; equal
action dimensions do not make old force checkpoints compatible. PPO is freshly
initialized. Explicit continuation is available only for a PVA checkpoint.

MPPI's iteration limit is zero, displayed as "No limit". Each window optimizes
until plateau or manual stop, with no wall-clock timeout. The first window
uses minimum20 iterations and patience15. Later windows require at least
3 iterations and3 iterations without meaningful
improvement (greater than max(0.1 score, 0.5% of the last improvement anchor's
magnitude)). The maneuver continues until a modeled hit, failure or its separate
5-second physical safety limit. That limit is editable and is not a computation
budget. Historical frozen runs retain their original stopping settings.

PPO defaults retain the flown neighborhood: start [-2,0,1.255] m, target
[-1,0,1.1] m, one second, each launch/target radius 5 cm. On9 September the
user changed MPPI to these same central positions, then explicitly clarified
that the horizon means rolling MPPI lookahead, not the entire whip. Following
authorized comparisons, the active lookahead is now2s. MPPI uses exact
initial/target coordinates (zero randomization radius).
Earlier MPPI artifacts preserve their original [0,0,1.225] → [1.5,0,1.1] setup.
These different tasks are not a controlled PPO-versus-MPPI benchmark.

The return rewards decreasing closest tip distance, near-target directed speed
and a valid tip-first strike. It charges time, displacement, jerk and failures.
The first valid modeled hit terminates reward accumulation; otherwise the
configured maneuver duration applies. The physical prediction continues to the next
30 Hz boundary for recovery handover. Invalid commands are rejected, never
clipped independently into inconsistent P/V/A. Feasibility limits remain
provisional model-envelope checks, not identified actuator limits.

## Fresh preliminary model

This fit stopped during full-whip cable-NN fitting at update 105, before plateau.
Weights and optimizer state are retained; the description below records its method.

`runs/adaptation/20260909-pva-M0-bootstrap` uses only five normalized cf7/adp0
flights and their actual command receipts. Original OptiTrack/controller CSVs,
the flown command, stored time alignment and stored Z normalization are frozen.
Normalization is applied once to measured drone and cable Z. Controller state
is not the fitting target. Geometry and measured 157 g drone + 18 g cable
provenance remain distinct from the later 153 g cf3 rig.

Cold nominal estimates and zero-output NNs replace old learned priors. Weak
regularization and broad parameter bounds are explicit. The drone nominal
response is fitted first, then its motion residual, then its attitude response
with the fitted translation frozen. Cable EI/Cb use bounded least squares on
the full-whip forward model. Two coefficients make finite-difference solves
much cheaper here than long autograd, without changing the forward equations.
This physical-fit window is 1.02 s from each last pre-onset sample, ending
about 11–19 ms after the one-second whip; its few boundary samples are retained
in the frozen physical-fit record. The NN objective below uses an exact mask.

An initial short-window cable-NN attempt lowered its own objective but worsened
complete-whip prediction. It is preserved and superseded. The stopped cable-NN
stage uses full-whip BPTT with loss exactly over 0–1 s; padded output samples outside
that interval have zero weight. Measured attachment motion isolates cable
identification. The final combined check uses predicted attachment motion and
no measurement corrections after initialization.

Stopping retains best weights and distinguishes practical plateau from a
safety ceiling. Drone: check every 10 updates, minimum 80, 5 stale checks,
0.5% meaningful improvement, ceiling 2000. Full-whip cable NN: check every 3,
minimum 12, 4 stale checks, 0.5%, ceiling 120. These are engineering rules, not
proof of a global optimum. Physical solver convergence and active bounds are
saved explicitly. The candidate becomes selectable only after essential checks.

The five takes repeat one local maneuver. This is an in-sample retrospective
bootstrap, not independent evidence that adaptation improves flight. Stored
post-hover normalization uses future samples and does not calibrate thrust.
New flights on one consistently configured drone provide prospective evidence.

## UI and export

Models & fitting owns model selection and current bootstrap jobs. Recordings
owns paired files/alignment. PPO and MPPI each expose launch, target, task,
reward, limits, optimization parameters, run library, plots and rehearsal/export.
Rehearsals provides a shared saved-result browser and a separate historical
force-policy view. Flight comparison uses normalized measured geometry and the
exact preflight saved forecast, including for new PVA rehearsal folders.

The 3D reference is blue; predicted whip is orange; predicted recovery green.
Plots separate desired P/V/A and predicted drone motion. Tracking error is
computed against the held command. Individual PPO episode values are retained
in `episodes.csv`, and can be overlaid on the progress plots.
Orange curves show deterministic evaluation reward/success; automatic PPO
stopping uses that evaluation reward, while blue curves show training batches.

CSV export contains whip, continuous recovery and final hold. Near-settled
endpoints use a gentle local return, avoiding an unnecessary large turning loop.
The whip packets remain byte-for-value unchanged when recovery is appended.
Complete command feasibility and height are checked before a CSV is written.
PVA curved recovery filters both exact minimum and maximum polynomial height
before selecting a turn, avoiding needless rejection of another feasible curve.
Recovery predictions remain outside the empirically assessed whip domain.
New generation rejects incomplete recovery predictions; historical partial
ghosts retain their original validity markers. No missing prediction is
replaced by the command. Packages contain the exact CSV, original forecast, frozen model,
policy or MPPI plan, settings and originating source snapshot.

Execution remains: take off → hold 10 s at the saved start → execute the entire
30 Hz CSV → land. There is no new ROS flight sender in this repository.
