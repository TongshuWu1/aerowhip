# M2 aggressive MPPI development run

## Latest: contact-speed weight 1,600

**User selected this exact run for use.** The selected flight package is
[`20260910-211435-608306`](../runs/flight_packages/20260910-211435-608306/README.md).
The complete CSV and original forecast are frozen; the global flight-selection
pointer identifies this M2 run. M2 recording inbox and predeclared 3/2 roles are
prepared. Selection audit: `runs/audits/M2-flight-selection-20260910`.

The user requested a further increase in hitting-velocity reward. New job
`runs/mppi_pva/20260910-211435-608306`; audit
`runs/audits/M2-impact1600-20260910`. Read status; do not duplicate/restart.
**Completed:** 20 updates / 67.93 s, reward plateau. The exact previous 35 commands
were retained, with **4.90513 m/s** directed contact speed: **no further speed gain**.
The complete CSV and saved forecast are byte-identical to the 800-weight run.
The higher score is entirely the increased bonus. Independent replay, full recovery
and native rendered-frame checks passed; 2,372 protected files are unchanged.
The separate rehearsal is open paused at the beginning. See the
[1,600-weight result](../runs/audits/M2-impact1600-20260910/README.md).

The only numerical setting change from the 800-weight run is `reward.impact=1600`.
M2, 512 samples, 1.5 s horizon, seed, noise, contact priority, scale 4 m/s,
preferred-wave shaping and feasibility bounds are unchanged.

The previous 35-command M2 winner is retained as an editable initial candidate,
with ten zero-jerk tail guesses. The two M1/M0 timing/strength baseline files and
wave reference are byte-identical. This is sequential reward development with an
inherited candidate, not a matched-initialization reward ablation. Preflight
reproduced the prior 4.90513 m/s contact speed and checked its rescored objective:
1,892.43069 at weight 1,600. Higher score from the weight alone is not a faster hit.
Four existing impact-reward tests and the CUDA preflight passed.

Formula: `1600 * v_plus^2 / (16 + v_plus^2)` at successful feasible contact.
There is no new velocity cutoff or earlier-hit reward. Save the new plan/rehearsal
separately; preserve the 800-weight run, fits, flown CSVs, ghosts and selections.

## Previous: contact-speed weight 800

User-authorized on 10 September 2026: plan a more aggressive whip with greater
hit strength using the newly refitted model. Job:
`runs/mppi_pva/20260910-210545-160853`. Audit:
`runs/audits/M2-aggressive-mppi-20260910`. Read status before acting; no duplicate.

**Completed:** 26 updates / 85.11 s, practical reward plateau. The new motion hits
in the M2 model at 1.152791 s and uses 35 commands (1.166667 s). Forward contact
speed rises from the same-model M1-command baseline's 4.50946 to **4.90513 m/s**:
**8.77%** speed increase, **18.32%** in squared-speed proxy. This is a modest
modeled increase, not measured impact power. The full 10.3 s CSV includes recovery.
Independent score/contact-speed replay, full recovery and native live/rehearsal
checks passed; all 1,957 protected files remain unchanged. Read the
[result report](../runs/audits/M2-aggressive-mppi-20260910/README.md).
Rehearsal `20260910-210545-160853-M2-frozen-refit-v1-whip` is open paused at start.
No promotion or real flight followed. Do not rerun the completed worker.

The model is exactly `M2-frozen-refit-v1`, signature
`be4bd82c038dad1e2f1508babc9a1c3da121e1f428f2a375942b382bf6a7b0b9`.
The preceding identification protocol, weights and models are unchanged. M2's
mixed development prediction is known; using it in this planning experiment is
explicitly requested, not an automatic promotion or proof of physical accuracy.

## Fixed search and changed objective

- Start tracked origin: `[0, 0, 1.255]` m; target: `[1.25, 0, 1.0]` m, radius 5 cm.
- Offline MPPI: 512 random samples, four proposal groups, 1.5 s complete search
  horizon, 45 possible 30 Hz jerk actions; stop the whip on feasible tip contact.
- Editable timing, strength and ten control points; seed 657; existing noise,
  20 minimum updates and 12-update plateau patience. No iteration ceiling or
  wall-clock planning deadline. Horizon is simulated motion time, not compute time.
- Retain feasible-contact priority, preferred-fold shaping, geometry, model,
  workspace/command envelopes and recovery checks.
- Increase the prepared M2 contact-speed bonus from 400 to **800**:
  `800 * v_plus^2 / (16 + v_plus^2)`, where `v_plus` is forward tip speed
  interpolated at successful entry. Misses/infeasible trajectories get no bonus.
  Four m/s is a smooth scale, not a hard threshold. No earlier-hit reward is added.

This is a directed speed/energy proxy. The simulator does not compute measured
impact force, impulse or delivered power. A larger reward does not by itself
establish a harder strike; compare actual predicted contact speeds.

## Editable command proposals and comparison

Baseline 1 is the exact 36-command flown M1 plan `20260910-181929-716218`, padded
with nine zero-jerk tail guesses. Baseline 2 is the exact 34-command flown M0 plan
`20260910-022818-648386`, padded with eleven zero-jerk guesses. Every action can
change. Both are newly simulated under M2; historical forecasts are not reused as
predictions. The existing preferred-wave shape reference is retained as an
objective reference, with its source/hash recorded.

The preflight found both command proposals feasible under M2. The M1 commands
predict a hit with forward tip speed **4.50946 m/s** and closest distance 4.436 cm.
The M0 commands predict a miss, closest distance 10.906 cm; their near-target speed
is not a hit-speed result. Evaluate the optimized candidate against these same-model
baselines. M1's original forecast predicted 3.94432 m/s at contact, but comparing
that number directly with M2 confounds model and command changes.

## Output and checks

Sixteen existing impact-reward, MPPI trajectory and timing tests passed.
Preparation binds the M2 candidate and its two learned residuals, settings, seed
commands and reference shape. The worker runs the frozen source snapshot on CUDA,
then independently replays the selected plan and appends/checks full recovery.
Live candidate geometry is bound to this job. The preview opens the saved rehearsal
when complete. Read the audit result for success, speed, stopping and preservation.

Original M0/M1 flight CSVs and prediction ghosts, M2 fit evidence, global planner
settings and flight/model selections are preserved. The new rehearsal is a separate
development candidate. No physical flight is dispatched by this job.
