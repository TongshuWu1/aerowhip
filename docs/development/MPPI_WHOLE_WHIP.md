# Offline complete-whip MPPI development trial

## Completed outcome

The trial finished at practical plateau after 20 updates in 63.28 s, including
the seed screen and independent final replay. Typical random-batch iteration was
about 2.85 s. The final selected minimum tip distance is 3.59 cm, but strict hit
success is FALSE and the dominant-bend detector completes 0/3 stages. This is
better target approach, not a verified whip-wave result.

Read-only analysis of the frozen forecast finds tip-first target entry at
1.12488 s. Tip forward speed 4.179 m/s, upward speed 2.854 m/s, direction angle
34.46 degrees; drone backward speed 1.298 m/s after 17.52 cm backward travel.
These pass the corresponding contact kinematics but do not satisfy the required
wave history. The detector is a kinematic proxy, not direct energy-flow evidence.

Full 1.5 s plan and 10.633 s command including recovery/final hold are saved in
`runs/rehearsals_pva/20260910-011618-458510-M0-development-whip`. Complete simulated
recovery passes the existing envelope; CSV peak speed 2.744 m/s and peak specific
force 16.650 m/s². Those bounds remain provisional, not measured hardware limits.
Selected score agrees with independent batch-one replay within 1.42e-10; rehearsal
cable prefix agrees within 7.86e-13 m. The original forecast manifest verifies
17 files. No later model was substituted and no real flight was performed.

The replay selection is Side XZ, quarter speed, at 1.12 s. Next inspection should
focus on whether the bend travels along the cable and on the upward component
at contact, without relabeling target contact as a strict successful whip.

User authorized 10 September 2026 after the receding search improved reward
while moving its selected prediction away from the target. The preceding run
`20260910-004200-654955` was stopped at 42 commands / 1.4 simulated seconds;
its original model, settings, search history and partial plan remain preserved.

New job: `runs/mppi_pva/20260910-011618-458510`.
Read its status and `runs/audits/mppi-whole-whip-20260910/status.json` for current
results. The live window follows this new job. Only one trial is authorized;
the two-update, 16-sample smoke check is a numerical execution check.

## Model and task

Same preliminary development M0: curvature regularization 2e-5, effective cable
velocity damping 0.4/s, cable NN disabled, original preliminary drone model and
residual. Source SHA256 d740595dc35f973e9ef2a927cf193bf2ace2612e4f4f4b8063d3ee5645755af0.
No fit, geometry, mass, controller or provisional envelope changes.
Start [0,0,1.255], target [1.25,0,1.0]. Model remains development-only,
fit_complete=false, flight_ready=false; not M1 or prospective flight validation.

The offline mode searches a complete 1.5 s maneuver from its initial state every
iteration. It does not commit a simulated 33 ms command then shift the horizon.
The separate return/recovery is appended and checked only after selection.
Strict contact, speed, direction, pullback and three-stage bend success checks
are unchanged. Continuous objective improvement is not a successful whip claim.

## Sampling and objective

- 512 random samples split over four maintained proposals, plus four deterministic
  means, best saved sequence and original editable native seed (518 batch rows).
- Ten XYZ latent control points, linearly interpolated to 45 commands and passed
  through tanh. The result is bounded jerk, integrated through the existing 30 Hz
  PVA contract and 150 Hz / eight-substep cable model. No state interpolation or
  prescribed cable motion replaces the simulation.
- A single initial screen uses diverse pull/release seeds projected into the same
  control-point basis; the native seed is the first 45 of the original editable
  60 commands, without time compression. Four distinct good control vectors
  initialize four independent exponentially weighted updates.
- Per-proposal temperatures target effective sample size of 20% of feasible
  random rows. Deterministic baselines do not enter weights. An entirely infeasible
  proposal retains its previous mean. Fixed latent noise mixture .03/.09/.2.
- Zero control prior. This is an adaptive-temperature MPPI-style trajectory
  optimizer, not exact importance sampling against a fixed Gaussian prior.
- At least 20 iterations, practical plateau patience 12, improvement threshold
  max(0.1, 0.5% of score magnitude). No iteration ceiling or wall-clock deadline.
- Preserve best complete feasible sequence across all iterations and prioritize
  strict successful candidates for final selection. Save means, RNG, best actions
  and stopping state each iteration. No automatic resume.

`trajectory_objective` is separate from legacy environment reward; PPO and older
MPPI modes retain their scoring. It combines closest-tip distance (weight 800),
continuous joint quality (400), small pull/release/bend credits (10/10/15), strict
hit bonus (4000), integrated soft vertical/approach/lateral costs (60/30/60), jerk
cost (.02), and exit speed/climb/forward-acceleration costs (.5/5/1).

Joint quality multiplies target proximity by smoothly clipped outward tip speed,
drone backward speed, backward travel and cable outward reach. Each motion factor
has a 0.2 floor, so partial progress has a useful score before binary success.
It has no wave-stage gate. Shape guidance is sampled at 30 Hz; closest distance
and strict strike events use the existing physics-rate checks. Scores log every
component, the legacy reward, failures, ESS per proposal, temperatures and hits.

The score remains an engineering hypothesis. A close target approach or high
joint score can still lack the required travelling bend. Inspect those separately;
do not weaken success labels or describe an ordinary swing as verified whipping.

## Verification and saved forecast

Twenty-one focused sampling, optimizer and development-model tests passed;
thirteen live-view and export tests passed on Windows / RTX 4080.
The two-update GPU smoke check compared selected batch and independent batch-one
scores within 1.44e-10, retaining model/asset hashes. It is not a success benchmark.
The full trial independently replays its final selection before marking its plan
complete. Source is frozen in the job; old forecasts are not regenerated.

The one-run supervisor then attempts a full rehearsal and recovery check. If it
passes, it freezes command CSV, model, prediction and source references in a
SHA-256 manifest for later prospective real-whip comparison. If it fails, it
preserves the result/error and does not restart, refit or fly automatically.
Evaluate future real takes against that original forecast before fitting M1.
