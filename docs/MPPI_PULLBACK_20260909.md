# Forward-pull / backward-release whipping task

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
