# Preferred-fold MPPI objective — 10 September 2026

## Current clarification: harder impact, not earlier time

The user corrected the earlier request: reward a harder tip hit. The earlier-hit
job `20260910-135950-415845` was cancelled while still prepared; its optimizer
never started. Historical earlier-hit support remains optional with missing
weight zero. Future MPPI settings now contain `impact=200` and
`impact_scale_m_s=4`, with the earlier-hit fields removed. PPO settings remain
unchanged. One corrected rerun is recorded under
`runs/audits/mppi-harder-hit-20260910`; read its status for the result.

Completed corrected run `20260910-140256-116887`: 20 updates in 64.87 s retained
the exact original 34-command motion. Directed impact speed remains 4.17886 m/s;
score 799.5256 includes 104.3717 bonus, with zero improvement over the rescored
baseline. A separate rehearsal and full recovery passed; its complete CSV is
byte-identical to the original selected M0 CSV. No harder motion was found.
The original flight remains selected; no automatic second trial is running.

At the actual successful interpolated tip entry, let
`v = max(0, dot(tip_velocity, desired_strike_direction))`. The added reward is
`200 * v² / (4² + v²)`: 40 points at 2 m/s, 100 at 4 m/s, and 160 at 8 m/s.
It remains increasing above the old 4 m/s shaping threshold. This is a bounded
directed kinetic-energy proxy, not measured force, impulse or collision energy.
No tip mass, target stiffness or contact duration is fitted. Extra transverse
or vertical speed alone earns no extra impact credit. The speed scale is not a
minimum-success gate or a cap. There is no explicit time preference.

The successful-hit velocity is latched independently of the existing first-any-
contact/closest-approach encounter used for fold shaping. Thus a different cable
node touching earlier cannot substitute its event for the tip's actual impact.
The state is included in receding branches and CUDA tick snapshots. Credit is
once per successful feasible episode, with invalidation removing credit; misses
and numerical/feasibility failures earn none. Historical missing weights yield
exactly the original score. All other objective terms and limits are unchanged.

Checks: 28 tests passed and one unavailable historical-fixture test skipped.
An actual Windows/RTX 4080 fixed-command check verifies fused/eager parity,
independent impact-velocity interpolation and unchanged old terms/motion. The
old selected motion has directed impact speed 4.17886 m/s and receives 104.3717
extra points. This rescoring is not evidence of a stronger new maneuver.

**Outcome: implementation and single trial completed; desired strong fold still
not achieved.** Job `20260910-015017-245988` stopped at practical plateau after
31 updates / 94.51 s. Closest modeled tip distance is 1.63 cm; first tip entry
into the 5 cm target is at 1.19697 s. Strict success is false. The full rehearsal
and recovery passed the existing simulation envelope and are frozen/selected
for Side XZ replay at quarter speed, 1.19 s.

| Motion feature before/at first encounter | Previous sweep | New trial | Preferred old |
|---|---:|---:|---:|
| Peak total turning [rad] | 1.600 | 1.634 | 3.054 |
| Peak tangent opposition | 0.023 | 0.026 | 0.906 |
| Forward tip reach [m] | 0.541 | 0.500 | 0.713 |
| Tip velocity elevation [degrees] | 34.27 | 17.04 | 17.09 |
| Shape-sequence similarity, full saved trace | 0.334 | 0.384 | 1.000 |

The tip direction is flatter, but folding only changes slightly and reach
decreases. This is **not an overall successful improvement**. Fixed diagnostic
task terms favor the previous sweep (706.6) over the new trial (661.3), so this
search did not even beat that known candidate under the new task terms. The
trial retained the exact old-command proposal, not the previous sweep as an
additional optimized baseline. That is a search-initialization limitation; it
does not justify claiming convergence to a global optimum or blaming the model
alone. The new sequence still lacks the old coherent loop. Its directed tip
speed is also below 4 m/s at first contact. Only one old wave stage is complete
before contact; the reported terminal 2/3 stages include later motion and must
not be interpreted as precontact wave propagation.

Independent selected-score agreement is 4.26e-10. Rehearsal cable-prefix
agreement is 1.52e-12 m. Whip duration is 1.5 s; full CSV including recovery/hold
is 10.6333 s. All 23 frozen forecast/provenance files and 21 protected prior
files verified unchanged. Audit `motion_review.json`, `contact_review.json` and
`motion_comparison.png` preserve the comparison. The displayed shape similarity
uses full saved 150 Hz traces; optimization uses 30 Hz shape snapshots, giving
a small numerical difference (new fold contribution 225.89 versus 225.73).

All optimization is now stopped/completed. No second trial or fit started.
A next authorized search should retain both known command baselines and test
whether the proposal representation can reach stronger preparatory folds before
another weight campaign. Preserve the current model and prospective evidence
sequence; neither this result nor the preferred old simulation proves real
whip behavior.

Post-run code review fixed one causal edge case: a contact during forward
carrier motion must not count the later end-of-tick forward peak as backward
travel. The cache now uses the preceding peak plus the interpolated encounter
position. This did not affect the selected reversing contact: a fixed two-row
CUDA replay agrees with the frozen trial score within 4.95e-11. A new targeted
test passes (26 in the first group, 22 related checks, 4 historical skips).
The original job source and all 23 forecast files remain unchanged; the current
source hashes and regression evidence are in `postrun_peak_regression.json`.

The user authorized implementation, fixed-motion preference checks and one new
MPPI trial after the [literature and archived-motion review](WHIP_WAVE_REWARD_REVIEW.md).
This changes the planning objective and editable proposal representation. The
development M0, launch geometry, strict historical hit criteria and provisional
physical bounds are unchanged. No fitting, controller change or real flight ran.

## Objective and event semantics

`preferred_fold_v1` uses bounded contact, shape-sequence and outward-cast terms:

`450 * contact + 600 * fold * proximity + 200 * cast - 300 * (1-proximity)`

It retains modest integrated carrier-approach/lateral penalties and the previous
jerk/recovery-end costs. The previous vertical-excursion penalty is zero in this
version; there is no new height bound. Historical wave-stage rewards and the
large binary success bonus are zero in this objective. Strict success is still
computed and reported with its original meaning, but does not override the new
scalar score in candidate selection. Older run scores/results are unchanged.

At each existing 150 Hz physics tick, swept-segment intersection finds the first
contact of **any cable node**, including the attachment. Tip position/velocity,
carrier position/velocity, backward travel and encounter time are interpolated
at that same fraction. This cache freezes at first contact. Without contact, it
keeps the closest swept tip approach and all corresponding kinematics. Later
near-target states cannot replace an invalid first encounter or improve its
contact/cast credit. This is the simulator's segment-interpolation convention,
not exact continuous nonlinear dynamics between ticks.

Proximity is `exp(-(tip_distance/0.35 m)^2)`. Contact quality multiplies proximity
by continuous forward-tip-speed, carrier-reversal and backward-travel factors,
each `0.2 + 0.8 * clipped_ratio`, using existing task thresholds. First non-tip
contact receives zero contact/cast credit. Cast additionally favors forward
reach divided by cable length and squared forward velocity alignment. Thus
vertical/lateral tip speed is softly discouraged near encounter, while the
preparation can move vertically to form the cable fold.

The shape prior uses the original liked farther-target simulated shape history
through its first tip encounter. It removes attachment translation, divides by
cable rest length, and preserves world strike-axis orientation. It compares
positions and local unit tangents at 21 rest-arclength locations and 21 temporal
phases. Tangent evolution represents spatial bending without a maximum-angle
threshold or a single argmax tracked between three cable regions. This is a
shape-sequence preference, **not a newly validated physical wave detector**.

Three monotone timing maps (`phase^0.75`, `phase`, `phase^(4/3)`) allow limited
timing variation, cover the complete sequence, and carry a small warp penalty.
Mean squared vector errors enter an exponential with position scale 0.12 cable
lengths and tangent-vector scale 0.55. Candidate shape capture is 30 Hz, ending
at the exact interpolated physics encounter; samples after the encounter are
excluded. A static fold cannot choose only one matching reference frame.
These are explicitly chosen preference parameters, not identified cable physics.

## Fixed checks before search

All results are in `runs/audits/mppi-preferred-fold-20260910`.

| Saved motion/control | Shape quality | Task terms, excluding command/recovery costs |
|---|---:|---:|
| Preferred old farther target | 1.000 | 1140.7 |
| Preferred old nearer target | 0.896 | 1090.1 |
| Previous new-M0 sweep | 0.334 | 706.6 |
| Exact old commands under new M0 | 0.443 | 586.1 |
| Static folded shape | below 0.000001 | -271.7 |
| Time-reversed shape sequence | 0.000183 | -259.3 |
| Shuffled shape frames | 0.00957 | 447.1 |
| Broad weak bend | 0.0147 | -267.1 |
| Rigid straight-cable swing | 0.0990 | 365.4 |
| Original fold, target moved 2 m sideways | 0.820 | -300.0 |

The counterexamples are deliberately synthetic objective tests, not feasible
drone commands or real observations. Reference self-similarity being 1 is
expected, not independent validation. The second liked motion provides a nearby
preference check, also from the old simulator. All ranking examples were fixed
before the new optimization; they do not prove that the new model can produce
the preferred motion or that the objective rejects every possible exploitation.

Translation changes the fixed task score by 2.3e-13. Subdividing every cable
edge leaves shape quality unchanged. Uniform 12% time stretching leaves shape
quality unchanged; velocities are correspondingly scaled for contact diagnostics.
Unit tests also check the contact latch, consistent interpolation, exclusion of
post-contact shapes, non-tip contact and horizontal velocity preference.

On Windows / RTX 4080 / float64, two complete CUDA-graph candidate rollouts
(old commands and the known current contact) agree with separate batch-one
traced replays: maximum score discrepancy 3.13e-9, maximum encounter-state
discrepancy 2.26e-11. The known contact remains at 1.1248781338 s. A separate
bounded two-update, 16-sample smoke check passed before the one trial.
Focused tests: 25 passed; related planner/wave/replay/export checks: 22 passed,
4 skipped because their archived historical fixtures are no longer active.

## Single trial and preserved provenance

Prepared job: `runs/mppi_pva/20260910-015017-245988`.
Both its `status.json` and the audit supervisor `status.json` are completed.

- Same development model source SHA-256
  `d740595dc35f973e9ef2a927cf193bf2ace2612e4f4f4b8063d3ee5645755af0`.
- Start `[0,0,1.255]` m; target `[1.25,0,1.0]` m, radius 5 cm.
- Whole 1.5 s maneuver, 512 random samples, four proposals, ten XYZ residual
  control points; adaptive effective-sample fraction 0.2, zero control prior.
- Exact 39 accepted commands from archived `20260909-173305-488091-mppi-wave`,
  followed by six zero-jerk commands. Their timing is preserved at zero residual.
  The old accepted forecast/model is never substituted for a new prediction.
- Perturbations interpolate in latent jerk space around those exact commands.
  A native-centered proposal is retained alongside three screened proposals.
- Minimum 20 updates, practical-plateau patience 12; no planning deadline or
  iteration ceiling. Same live 3D snapshots and independently checked recovery.

The job freezes the shape reference and its source hash, initial commands and
their provenance, model assets, settings and source snapshot. The supervisor
runs once, attempts rehearsal/recovery and freezes a forecast manifest. Failure
or stop is preserved without automatic restart, alternate run or fitting.

The next physical evidence remains prospective: freeze the forecast, collect
new real whip data, compare with that original forecast, then review the data
before an M1 fit. This shape prior does not turn preliminary-only development
into an M1 model or establish flight readiness.
# Later reward addition: earlier successful hits

**Historical interpretation, superseded by the harder-impact clarification above.**
Its prepared rerun was cancelled before launch; the time bonus is no longer in
future MPPI defaults. The following records the earlier implementation decision.

On 10 September 2026, the user requested a preference for faster target hits.
Future MPPI defaults and `config/pva/mppi.json` now set `reward.early_hit=100`
and `reward.early_hit_scale_s=1`. The additional term is
`100 * exp(-actual_successful_hit_time_s / 1)` once per successful feasible
episode. A 0.8-second hit earns 44.93 extra points; a 1.2-second hit earns 30.12.
Misses, failures and invalid times earn zero. It is a soft preference; other
reward terms can still outweigh it. There is no new duration/acceleration limit.

The time is the interpolated tip-entry time measured from maneuver start, not
optimizer wall time, lookahead length or recovery duration. The fixed decay scale
does not change with horizon. Streaming credit is added as a difference so it
cannot repeat after a hit; whole-trajectory scoring adds the same term explicitly.
Missing keys retain the original score, including the selected M0 and PPO runs.
The Rewards UI exposes the maximum bonus and decay time in seconds.

No new optimization or flight was launched. The saved M0 flight remains frozen.
Audit `runs/audits/mppi-earlier-hit-20260910`: 23 tests plus a Windows/RTX 4080
fixed-command check confirm +32.4692 points for the selected 1.124878 s hit,
identical motion/outcome and unchanged other terms. This is a new diagnostic
score, not a replacement for the historical saved score. PPO settings unchanged.
