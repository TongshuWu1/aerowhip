# MPPI timing and strength search — 10 September 2026

**Completed: no improvement over the previous sweep.** The one trial stopped
at practical plateau after 20 updates / 64.40 seconds. Best score is 696.013,
exactly the retained previous-sweep baseline. The final 45 command vectors are
byte-for-byte numerically identical to that baseline's saved array; they were
reevaluated with the frozen development M0, not replaced with an old forecast.
The initial samples and all new search evidence are preserved.

Closest modeled tip distance remains 3.59 cm; first tip entry is at 1.124878 s,
tip velocity elevation 34.27 degrees, forward reach .541 m and peak total cable
turning 1.600 rad. The strong fold remains absent and strict success is false.
This is not a successful new maneuver, physical validation, or an M1 result.

Only 123/512 initial candidate rows passed the modeled feasibility checks
(including four exact zero-parameter rows). During the 20 updates, on average
71.25% of random candidates failed those checks (range 68.36–79.49%). This shows
that much of the broader timing/strength exploration was wasted on infeasible
motion; it does not identify the individual active constraints or prove that
the desired fold is physically impossible. The initial screen's best feasible
shape similarity near the target was .443, still far below the liked reference.
See `search_review.json`; these diagnostics read saved arrays without more runs.

Independent selected-score agreement is 2.02e-10. Rehearsal and recovery passed
the existing simulation envelope and are saved at
`runs/rehearsals_pva/20260910-020448-013542-M0-development-whip`.
All 26 frozen files and 47 protected prior files verified unchanged. The new
rehearsal is selected for Side XZ replay at quarter speed / 1.12 s and explicitly
labeled as a retained baseline, not an improved whip. Optimization is stopped;
no further trial, fitting or flight started.

A next investigation should attribute sampled failures to individual existing
constraints and test whether the desired preparatory fold is reachable under
the fitted command-to-motion model. The present trial does not support merely
adding more samples, loosening physical bounds, or claiming that tuning the
reward has solved the wave problem. Any subsequent campaign needs a new scope.

The user authorized one controlled trial after the preferred-fold reward trial
failed to beat a known earlier new-M0 candidate. This changes the search
representation and initialization, not the reward or model.

Job: `runs/mppi_pva/20260910-020448-013542`.
Audit/supervisor: `runs/audits/mppi-timed-baselines-20260910`.
Check their status before any subsequent action; no automatic alternate trial.

## Search change

Both exact command baselines are evaluated in every batch:

1. The liked archived 39-command maneuver, padded with six zero-jerk commands,
   evaluated under development M0. Commands are copied from the preceding
   reviewed proposal; its complete archive provenance remains included.
2. The 45-command new-M0 sweep from `20260910-011618-458510`.

There are four proposal distributions, with baseline families `[0,1,0,1]`.
Two start centered on the exact baselines, and two start from the strongest
screened sample in their respective family. Every update evaluates 512 random
samples, four proposal means, the incumbent and both exact baselines (519
candidate rows). Deterministic rows never enter the exponential weight update.
The incumbent's objective cannot fall below either reevaluated baseline.

Each proposal has ten XYZ smooth residual control points plus nine new scalars:
two duration logits and one strength parameter for each of X/Y/Z. Three ordered
source phases have boundaries `[0, .45, .90, 1.50]` seconds. Softmax duration
fractions define an independently editable monotone time map for each axis,
covering the complete source sequence in the same 1.5-second destination.
These are command-time landmarks, not measured physical phase boundaries.

The retimed 30 Hz command is the **interval average of the source zero-order-held
jerk**, integrated separately across each timing phase. This retains short pulse
contributions when transitions move between command knots. Point-sampling a
retimed pulse could otherwise miss it. The per-axis strength multiplies its
latent jerk amplitude, and smooth residual controls remain additive in latent
space. Tanh enforces the existing normalized jerk range. PVA commands are always
reintegrated and checked; no old state or predicted trajectory is substituted.

Timing noise standard deviations are `[.12,.30,.60]` in duration-logit units;
strength noise `[.10,.20,.40]` in log-gain units, paired with existing smooth
control noise `[.03,.09,.20]`. Timing logits are numerically clamped to ±8 and
log strength to ±4 to prevent degenerate numerical intervals/gains. These are
search-parameter safeguards, not measured vehicle bounds or new height limits.

Unchanged: development M0 SHA
`d740595dc35f973e9ef2a927cf193bf2ace2612e4f4f4b8063d3ee5645755af0`,
`preferred_fold_v1` reward, 1.5-second complete maneuver, 512 random samples,
four proposals, adaptive effective-sample fraction .2, zero control prior,
20 minimum updates/12 plateau patience and no planning deadline/iteration cap.
Start `[0,0,1.255]`, target `[1.25,0,1.0]`, radius .05 m. Strict historical strike
criteria and all provisional physical bounds remain unchanged. The previously
verified causal encounter-peak fix is included; no fitting or flight ran.

## Checks and provenance

44 focused tests passed: identity preservation of both baseline families,
sub-command pulse integration, independent timing/strength edits, boundedness,
batch/single-draw agreement, deterministic sampling, configuration checks,
existing objective/optimizer tests, and related live replay/export checks.

A bounded two-update / 16-sample GPU smoke check passed on Windows / RTX 4080 /
float64. Its independent batch-one score differed by 7.08e-11. Baseline scores
are 571.067 (old commands under M0) and 696.013 (previous sweep), including all
unchanged objective costs. The exact better baseline was retained in the smoke.

The job freezes the baseline command array, complete source provenance, shape
reference, model assets, settings, code snapshot, baseline evaluations and
optimizer checkpoint. The one-run supervisor attempts checked rehearsal/recovery
and freezes a forecast manifest. Old raw data, models, commands and forecasts
remain unchanged. This is a simulation search test, not physical validation or
evidence of M1 adaptation.
