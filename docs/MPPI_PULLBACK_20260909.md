# Forward-pull / backward-release whipping task

## Farther and lower target trial

The user requested a farther and slightly lower target. Trial
`20260909-172921-966356` moves the target from [-1,0,1.1] to [-0.75,0,1.0] m:
horizontal start-to-target distance increases from 1.0 to 1.25 m and target
height drops by 0.10 m. The tracked-origin start remains [-2,0,1.255] m.
Reward, model, task qualifications, limits and 2 s/1,024-candidate sampling are
unchanged from the accepted outward-cast configuration. The previous accepted
actions initialize a fully editable proposal in a fresh frozen job; old commands,
forecasts and targets are not translated or relabeled. PPO and fitting stay stopped.
Audit: `runs/audits/mppi-farther-20260909-172921-966356`.

Accepted full candidate `20260909-173305-488091` independently passes the hit,
recovery and export checks. Parent stopped at 42 iterations (238.69 s) and four
committed actions; its rolling loop did not finish. The derived job performs no
new optimization and inherits warm-start search cost. Hit time is 1.288116 s,
minimum tip distance 4.1154 cm, wave completion 1.253333 s; 39 commands end the
whip at 1.3 s. Full CSV/recovery lasts 10.433333 s. Exact portable replay passes
for CSV bytes and all 12 arrays, Windows/RTX 4080. This is a target-only trial;
existing reward/model/limit/sampling settings and PPO settings were checked unchanged.

| Metric | Previous target | Farther/lower target |
| --- | ---: | ---: |
| Forward cable reach at contact | 0.716519 m | 0.712673 m |
| Forward drone displacement at contact | 0.218577 m | 0.474605 m |
| Peak forward drone displacement | 0.588776 m | 0.739561 m |
| Sampled peak sideways displacement | 0.033162 m | 0.110629 m |
| Distal RMS elevation at contact | 51.93° | 47.74° |
| Tip velocity elevation at contact | +17.07° | +17.09° |

The farther target is reached mostly through increased carrier approach, not
more cable extension. At contact the drone retreats at 1.469089 m/s and the tip
advances at 4.008100 m/s: only 0.0081 m/s above the unchanged speed requirement.
Do not claim a robust hit or a solved horizontal whip. Same historical normalized
M1 simulation only. Active MPPI target and selected exact rehearsal now use the
new result, Side XZ/quarter speed/1.03 s. Previous rehearsal and all original
commands/forecasts remain intact. No worker remains and no restart is scheduled.
Accepted audit, animation, package, verification and adoption snapshots are under
the trial audit's `accepted` directory.

## Outward cast objective

The user clarified that the intended whip unfurls toward a target beyond the
carrier's approach, rather than bringing the drone close and swinging down or
sideways. Horizontal tip velocity alone does not capture that task.
The saved horizontal candidate reaches forward only 0.420545 m from its actual
rotated attachment (44.15% of 0.9525 m rest length), with 0.730285 m peak carrier
advance toward a target initially 1 m ahead. Its contact tracked origin is
[-1.484455, 0.471579, 1.796314] m, revealing the sideways component.

Authorized trial `20260909-170755-303192` adds four optional soft terms in
`learning/horizontal_strike.py`. Forward reach is the attachment-to-tip vector
projected on the requested strike direction, divided by structural rest length
and clipped to [0,1]. Carrier translation cannot improve this feature.
Contact receives 1000 times squared reach; near the target after forward
preparation, the running cost is 120 times squared reach shortfall times the
existing Gaussian proximity factor. Forward drone displacement and horizontal
sideways displacement receive squared running costs of 180 each (points/m²/s).
The previous horizontal-contact bonus is reduced from 320 to 80; previous
vertical excursion/alignment/velocity costs stay 80/80/40. These preferences
allow folding during preparation and do not impose a straight cable throughout.

Same start, target, historical normalized M1, task checks, physical limits,
two-second lookahead and 1,024-sample mixture. No new hard height or reach bounds.
Initialization is the previous accepted sequence with every action editable.
PPO and fitting remain stopped. Do not compare raw scores across objectives or
claim a travelling-bend proxy proves mechanical energy transfer. Compare actual
forward extension, carrier approach, top/side geometry, speed and modeled hit.
The audit now includes `cast.png` to expose sideways motion and plot cable reach
against carrier advance. Targeted tests: 22 passed, Windows/RTX 4080.

First trial stopped at 31 iterations/three committed commands. Its saved full
proposal reaches 0.553195 m forward (58.08%), versus 0.420545 m before; peak
carrier approach is 0.693238 m and sideways excursion 0.249497 m. Contact velocity
elevation worsens to +20.65°. Refinement `20260909-171100-183874` keeps the objective
but changes the sampling mixture from [.05,.15,.35] to [.01,.03,.09], since broad
noise rejects approximately 95% of trajectories and rarely samples another hit.
The entire proposal remains editable. This changes sampling, not physics or task
qualification. Per-sample nominal hits are not policy success probabilities.

The fine-sampling parent stopped at 69 iterations/four committed commands. Its
latest saved proposal reaches 0.713970 m (74.96%), with 0.610556 m peak drone
advance and 0.096585 m sampled sideways excursion. Contact tip elevation is
+31.42°, so the reach improvement carries an upward-velocity tradeoff. Balance
trial `20260909-171728-339853` increases vertical excursion cost to 300, vertical
tip-velocity cost to 160 and horizontal contact bonus to 320. Reach weights and
all hard limits stay unchanged. It starts from that full editable proposal.

The audit renderer now appends the interpolated contact endpoint and holds it
briefly in animations; previously, the sampled animation could stop visibly
short of contact. It reads saved arrays and does not regenerate old forecasts.
The independent bend-stage diagnostic still uses the pre-contact sampled history.

### Accepted outward-cast improvement

Accepted full candidate `20260909-172103-268459` is a separate derived job from
the balance trial's saved proposal. All three parents are stopped: 31, 69 and
45 iterations, with 3, 4 and 4 committed commands respectively. Their receding
loops did not finish. Independent full-proposal replay, recovery and package
verification passed. Do not report the derived job's zero new iterations as a
zero-cost solve; these trials consumed approximately 793 s of optimizer time,
plus the earlier warm-start searches and separate validation/export work.

| Metric | Previous horizontal rehearsal | Accepted outward cast |
| --- | ---: | ---: |
| Forward reach from attachment at contact | 0.420545 m (44.15%) | 0.716519 m (75.23%) |
| Peak forward drone displacement | 0.730285 m | 0.588776 m |
| Forward drone displacement at contact | 0.515545 m | 0.218577 m |
| Sampled peak sideways drone displacement | 0.465441 m | 0.033162 m |
| Pre-contact peak drone height | 2.188726 m | 2.105445 m |
| Tip velocity elevation at contact | -1.26° | +17.07° |
| Distal cable RMS elevation at contact | 51.82° | 51.93° |

The new hit occurs at 1.234274 s (minimum tip distance 3.3087 cm), after bend
completion at 1.206667 s. At interpolated contact, drone forward velocity is
-1.905981 m/s and tip forward velocity +4.102264 m/s, with 0.370199 m backward
travel from the drone's forward peak. The 38-command whip ends at 1.266667 s;
complete recovery/CSV is 10.4 s. Same historical normalized M1, start and target.
Full package replay reproduces the CSV byte-for-byte and all 12 arrays exactly.
27 targeted reward/contract/planner/export/UI tests passed on Windows/RTX 4080.

This increases outward reach and removes most sideways swing, but still includes
substantial climb and a steep distal cable. It is not a solved horizontal whip,
proof of energy transfer, or physical flight validation. Impact forward speed is
only slightly above the unchanged 4 m/s criterion; no robustness claim is made.

`config/pva/mppi.json` now adopts the final weights and [.01,.03,.09] noise
mixture for future launches. No new hard bounds; task/launch/limits are unchanged.
PPO settings were checked unchanged by SHA256. The new exact rehearsal is selected
in `config/pva/replay.json` at 1.03 s, Side XZ, quarter speed. Previous artifacts
are preserved. Audit, motion, comparison views, package and adoption snapshots:
`runs/audits/mppi-outward-20260909-171728-339853/accepted`.

## Horizontal-strike reward trial

The user requested a more horizontal outward strike and explicitly rejected a
new hard height band. Trial `20260909-162950-671240` keeps every existing task,
feasibility limit, launch coordinate, model and sampling setting unchanged.
The previous wave sequence initializes an editable proposal; no prefix is fixed.
PPO and fitting remain stopped. Trial settings are frozen in its run directory;
the prior selectable result remains intact.

`learning/horizontal_strike.py` adds four optional, purely soft reward terms:
20 times squared vertical drone displacement per second; 40 times vertical tip
velocity fraction squared per second; 40 times mean squared vertical tangent
component of the last three cable segments per second; and up to 160 extra points
for horizontal contact. The latter bonus is the product of horizontal fractions
of distal alignment and tip velocity, evaluated at interpolated contact.
The two near-strike costs are multiplied by the existing Gaussian target-proximity
factor and require completed forward preparation. Vertical displacement remains
available at finite cost. None of these terms alters hit qualification or the
allowed workspace. GPU/eager parity preserves the previous modeled hit.

Two trials are now complete. First parent `20260909-162950-671240` was stopped
at 42 iterations/four committed actions. Its full proposal initialized the second
parent `20260909-163336-614991`, increasing vertical excursion to 80, alignment
to 80 and contact bonus to 320; velocity cost remains 40. Every action remained
editable. Second parent stopped at 40 iterations/three committed actions after
independent full-proposal strike/recovery/export acceptance. Neither rolling loop
completed. Two-second lookahead, 1,024 samples, existing convergence stopping and
all physical/task limits stayed unchanged; there was no fixed planning-time cap.

| Pre-contact/contact metric | Saved baseline | First trial preview | Accepted second trial |
| --- | ---: | ---: | ---: |
| Tip velocity elevation at contact | +18.04° | -9.02° | -1.26° |
| Distal cable RMS elevation at contact | 58.15° | 51.57° | 51.82° |
| Drone peak height | 2.327 m | 2.289 m | 2.189 m |
| Three bend stages before modeled valid hit | Yes | Yes | Yes |

Distal RMS elevation means asin(sqrt(mean squared vertical tangent component))
over the last three segments. Positive velocity elevation means upward motion.
This improves horizontal tip motion and modestly reduces climb; the cable is
still steep. It does not establish a horizontal whip or physical flight success.
Scores across these different reward weights are not comparable.

Accepted derived job `20260909-163646-375180` uses 38 actions (1.266667 s), hit
time 1.263487 s, minimum tip distance 3.7583 cm, and bend completion at 1.24 s.
Full recovery/CSV lasts 10.4 s. Rehearsal:
`runs/rehearsals_pva/20260909-163646-375180-mppi-wave`.
Audit/animation/package: `runs/audits/mppi-horizontal-20260909-163336-614991/accepted`.
The derived job does no new optimization; optimization cost belongs to both
warm-start parents and the original baseline search. Active MPPI defaults remain
unchanged. The user subsequently selected the accepted second
trial in Rehearsals (Side XZ, quarter speed, 1.03 s), preserving the baseline.
New controls default to zero in MPPI.
Targeted reward, wave, pullback, receding-horizon and export tests: 21 passed on
Windows/RTX 4080. No fitting/PPO/heartbeat was restarted.
Portable package replay reproduces the complete CSV byte-for-byte and all 12
saved arrays exactly; see `accepted/verification.json` in the audit directory.

## Current extension: travelling bend

The latest user request focuses on MPPI; PPO exploration/training is a separate
problem and remains stopped. Active `config/pva/mppi.json` now enables `require_wave`.
The original completed result below already has a bend moving toward the tip;
it was previously assessed only through carrier reversal and target contact.
The new objective explicitly measures stronger bending and ordered propagation.

`learning/whip_wave.py` measures the local turning angle between neighboring
segments and the material coordinate of its maximum. After forward pull has
qualified, that maximum must persist for 0.04 s in each ordered region: proximal
coordinate <0.45, middle 0.45–0.75, distal >=0.75. Active minimum local angles are
0.20, 0.35 and 0.65 rad respectively. Regions use rest-length coordinates from
attachment to tip. Completion must precede the contact interval; post-contact
samples cannot qualify an early hit. Three phase rewards total at most 120 and
cannot be collected again by oscillating. First-contact, speed/direction, pullback
and complete command/model envelope checks remain in force.

This is a **kinematic travelling-bend proxy**, not a proof of mechanical energy
transfer or a tapered-whip crack. It rejects a rigid swing and a stationary bend,
but coarse dominant-peak tracking cannot prove that every stage belongs to one
continuous physical wave. Thresholds are engineering choices for this cable mesh.
Inspect full shape sequences and tip velocities as well as the scalar acceptance.

New cold run `20260909-155433-585040` uses a 2 s lookahead and 1,024 samples plus
the deterministic mean, with native bounded jerk and unchanged physical limits.
Initialization screens 1,025 pulse sequences with varied preparation, reversal,
and vertical-lift timing. Equal-sized interleaved Gaussian groups use latent
standard deviations 0.05/0.15/0.35 with temporal correlation 0.9. This mixture uses
zero control prior; nonzero prior is rejected because the existing single-Gaussian
likelihood correction does not apply. Importance weighting is retained, not an
elite/CEM update. Later windows use minimum five iterations and four stale
iterations; first window minimum 20, patience 12. There is no fixed optimizer
time/iteration cap. The separate maneuver safety limit remains five seconds.
Closest-distance weight is reduced 60→20 and time cost 10→5 points/s to reduce
pressure against preparation. PPO settings, models and historical outputs are unchanged.

Inspect shapes without regenerating saved ghosts using `tools/audit_whip_wave.py`.
The baseline audit is `runs/audits/mppi-wave-baseline-20260909`; setup provenance is
`runs/audits/mppi-wave-setup-20260909`.

### Accepted wave plan

Completed derived plan `20260909-160208-467697` accepts the independently replayed
full proposal captured after four parent commands. It has 41 actions / 1.36667 s
and a modeled hit at 1.33338125 s, minimum distance 2.68814 cm. Ordered bend stages
complete at 0.81333, 1.15333, 1.30000 s. Peak local turning angle is 0.79806 rad
(earlier result 0.62109 rad); peak tip speed 5.78353 m/s. Drone backward speed at
contact is 0.62381 m/s, tip forward speed 5.21677 m/s, backward travel 0.132719 m.
The complete 10.5 s recovery/hold CSV passes the saved envelope and exact portable
replay; modeled drone peak height is 2.32653 m. This uses more height than the old
result and has no new physical validation.

The parent rolling run is STOPPED at 13 committed commands after 97 completed
iterations / 508.99 s. It first found a predicted valid wave hit at iteration 17 /
94.69 s. Its rolling loop did not finish. The accepted full candidate is a separate
result with explicit source/hash provenance; no new optimizer runs in that derived
job. Do not interpret its zero iterations as free planning or the parent stop as a
fixed planning-time cap. `tools/finalize_mppi_candidate.py` accepts only a replayed
valid hit that also passes complete recovery and export. It does not stop the parent.
The parent was explicitly stopped after these checks passed.

[Final audit](../runs/audits/mppi-wave-final-20260909/result.json),
[portable verification](../runs/audits/mppi-wave-final-20260909/verification.json),
[animation](../runs/audits/mppi-wave-final-20260909/motion/wave.gif), and
[shape/velocity plot](../runs/audits/mppi-wave-final-20260909/motion/wave.png).
28 targeted tests pass on Windows/RTX 4080, including wave ordering/persistence,
rejection of rigid/stationary bends, CUDA branch parity, contact causality, and
independent GUI settings. A stale legacy snapshot test was corrected to edit the
actual configured source task instead of assuming a hard-coded historical target.

## Historical hit-and-reversal task and result

The user clarified that a fast tip contact alone is insufficient. The intended
motion is forward carrier motion to load the cable, then backward carrier
motion while the cable continues forward toward the target. The previous1.2s
result20260909-131121-677417 had modeled forward drone speed~0.99m/s at contact
(peak~2.30m/s), and no commanded backward motion before contact. It remains
valid only under its original tip-hit task; do not call it the requested whip.

## Explicit measurable contract

New MPPI settings enable `task.require_pullback`. The modeled actual tracked
origin must first advance>=0.25m along the strike direction at speed>=1m/s,
then retreat>=0.10m from its running forward-position peak. At cable contact,
modeled drone velocity must be<=-0.5m/s along the strike direction. Existing
tip-first5cm sphere entry, minimum forward tip speed4m/s, angle<=45 degrees,
first-contact-only and command/model envelope limits still apply. These
editable thresholds make a genuine reversal measurable rather than accepting
deceleration or a small numerical sign change. They are engineering thresholds,
not an identification of a minimum cable energy or proof of wave propagation.

Bounded once-per-phase progress rewards20/40 favor pulling then releasing.
Repeated reversals cannot collect extra phase reward. Directed tip-quality
reward is gated by the pullback condition. A token reversal followed by renewed
forward drone motion at contact cannot satisfy the task. Desired command
reversal alone is insufficient: the fitted drone's actual motion is checked.

New phase tensors are copied by GPU tick graphs and candidate branching.
PPO settings and existing observation contracts remain unchanged when this
optional task is disabled. No PPO training or model fitting was restarted.

## Initialization diagnostics

Tests reject the old forward-only hit and check phase ordering, nonrepeatable
rewards and nonzero-time CUDA branch/eager parity.
Cold2s/2.5s candidate comparisons: runs/audits/mppi-pullback-windows-20260909.
Settings/provenance: runs/audits/mppi-pullback-setup-20260909. These diagnostics
are complete. Historical normalized M1 simulation only; the physical drone
investigation remains separate.

Both zero-initialized cold windows converged to misses without completing
reversal (2s31 iterations,2.5s38 iterations). A batched screen of321 bounded
30Hz jerk guesses (zero plus forward/negative/backward acceleration profiles
with different amplitudes, hold times and vertical lifts) found a completed
pull/reversal with no early contact and a5.12cm miss. This is an initialization
guess only; MPPI freely optimizes all jerk values afterward. No PPO checkpoint,
CEM elite update or spline action model was introduced.

The provisional full2s rolling diagnostic with this initialization is at
`runs/audits/mppi-pullback-rolling-20260909/result.json`. It froze code including
height-constrained recovery and saved modeled drone/cable velocities, which
exposed the contact-timing issue described below.

73 relevant tests pass on Windows/RTX4080/PyTorch2.11.0+cu128, including old
PPO/force recovery contracts and new ordered pullback/state/seed tests. Active
MPPI defaults now enable this task and the pullback initialization. Native app
was reopened. The library labels old tip-hit tasks separately, and new saved
velocity data enables a Pull and release view: drone/tip displacement, signed
velocities and the signed velocity pattern along the cable. No old ghost is
regenerated to create these plots. Final run/export/plot verification passes.

## Contact-time correction

Run20260909-133459-410882 completed a provisional hit at1.2666667s and exported
exactly. The velocity plot exposed an interval-timing mismatch: backward travel
was0.100015m at the physics interval end1.273333s, but contact occurred earlier
within that interval. The new criterion must hold at contact, not just by the
end of its interval. This provisional result must not be called a final strict
pullback result. Its saved files remain under their original code semantics.

New pullback hit checks interpolate drone position/velocity and cable-tip
velocity at the same sphere-entry fraction. Forward loading must already have
qualified. Regression rejects the old interval-end result. The valid first30
commands (1s, before any contact) are explicitly reused in a NEW continuation;
they are replayed and checked under current code before new release optimization.
Audit `runs/audits/mppi-pullback-contact-20260909/result.json` identifies the
completed continuation. Its source/identity record the exact parent hash and
prefix length; no old snapshot was changed. No live optimizer remains.

## Verified result

Final run `20260909-135429-797997` completes the corrected task. It preserves
the parent's first30 commands exactly and refines the final9 commands, taking
36 new optimizer iterations and56.934s. This is continuation timing, not a
cold-solve benchmark. Lookahead is2s/60 actions with1024 samples plus the mean;
one30Hz action is committed per replan. The separate maneuver safety limit is
5s; no optimizer wall-clock or iteration cap is imposed.

The modeled drone advances0.65475m, reaching +1.30192m/s along the strike axis.
At interpolated contact1.26666754s, it has retreated0.100000267m and is moving
-0.62040m/s, while the tip moves +5.02739m/s forward. Minimum tip distance is
0.0134697m. The ordered pullback check passes at contact. These are signed
motion measurements, not a quantitative energy-transfer or physical-flight
validation. Start[-2,0,1.255] and target[-1,0,1.1] remain PPO-matched.

The1.3s whip plus recovery/hold produces a10.4333s complete command. Saved
reference and modeled envelope checks pass: command Z1.255..2.13853m, modeled
drone Z1.255..2.06169m, minimum cable Z0.2475m. Streaming/export differences
are4.66e-15m for drone position and8.90e-13m for cable position. Portable replay
reproduces the CSV identically and every saved array with zero difference.

Artifacts:

- Final audit/package: `runs/audits/mppi-pullback-contact-20260909`.
- Rehearsal: `runs/rehearsals_pva/20260909-135429-797997-mppi-diagnostic`.
- Signed motion plot/metrics: `runs/audits/mppi-pullback-final-motion-20260909`.
- Native six-page UI/rehearsal/Pull and release audit:
  `runs/audits/mppi-pullback-final-ui-20260909` (no errors or started jobs).

The final relevant suite passed73 tests in30.95s. Regression includes rejecting
both the old forward-only hit and the provisional interval-end pullback hit,
ordered phase rewards, CUDA state parity and explicit prefix continuation.
PPO configuration, frozen model, physical limits and historical artifacts are
preserved. Fitting/PPO remain stopped and both heartbeats PAUSED.
