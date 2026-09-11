# Whip success condition: literature and code audit

10 September 2026. Review only, following the user's request to stop adding
complexity and examine the papers and success condition. No planner, reward,
model, settings, replay selection or historical result was changed. No physics
rollout, optimization or fitting ran.

**Finding:** our `success` boolean combines target contact with several
uncalibrated motion/style requirements. It is not a literature-established
definition of successful whip targeting. Saying simply that the recent runs
“failed to hit” would be incorrect: both reviewed motions made modeled tip
contact. They failed our custom composite test. They also differ visibly from
the preferred old fold; those are separate findings.

## What the papers actually assess

- **McMillen & Goriely, Whip Waves (2003):** studies cracking mechanics in a
  planar, inextensible, unshearable, tapered elastic rod. The forward-crack
  discussion concerns formation and travel of a loop, free-end acceleration,
  taper and applied tension. Crack production concerns supersonic motion, not
  reaching a target sphere. This motivates examining a travelling fold, but
  provides no basis for our particular 4 m/s, three-band angle/dwell or drone
  reversal gates. Our nonuniform marker-loaded aerial cable is a different
  system. [Publisher abstract/introduction](https://www.sciencedirect.com/science/article/pii/S0167278903002215)
  and [author-hosted paper, indexed conclusion](https://goriely.com/wp-content/uploads/2003-PhysDwhip-waves-1.pdf).
- **Edraki et al. (2025), Human-Inspired Robot Whip Manipulation:** equations
  3–5 use minimum squared tip-to-target error plus squared control effort.
  A distance below 1 cm counts as a hit; Figure 2 compares striking-only with
  preparation-plus-strike. Preparation improves range/effort, but is not itself
  required to declare a hit. There is no added bend-stage or tip-speed gate in
  that stated definition. [Complete paper, pp. 2–3](https://deformable-workshop.github.io/icra2025/spotlight/01_01_05_Edraki_Human.pdf).
- **Nah et al. (2023), Learning to manipulate a whip with simple primitive
  actions:** the detailed optimization section tracks 25 whip nodes and counts
  contact by any of them with a 3 cm-radius target sphere. Thus even “tip only”
  is not universal across the targeting literature. Its nine parameters describe
  one arm movement, not nine success conditions.
  [Optimization of the whip task](https://doi.org/10.1016/j.isci.2023.107395).

Source-access limit: the 2003 full PDF download failed (web timeout/local
certificate errors). Its publisher text and indexed author-hosted passages were
read, including the conclusion; this is not a claim of a fresh page-by-page
reading of the entire 34-page PDF. The complete four-page Edraki PDF was read;
the 2023 optimization/results sections were available as indexed primary text.

## Current code condition, exactly

`learning/pva_env.py` and the frozen latest `settings.json` require all of:

| Gate | Active value | Assessment |
|---|---|---|
| Tip enters target sphere | radius 5 cm | Geometric task choice; radius must follow the actual target/protocol |
| Tip precedes other cable nodes at first contact | Strict first-contact-only | An additional task choice, not universal in the papers |
| Tip velocity projected on world +X | at least 4 m/s | No measured impact requirement or cited calibration establishes this value |
| Full 3D velocity direction | within 45 degrees of +X | Direction preference, not a targeting definition from these papers |
| Prior drone forward motion | displacement ≥25 cm and speed ≥1 m/s | Prescribed strategy with our numerical thresholds |
| Drone reversal at contact | backward travel ≥10 cm and speed ≥0.5 m/s | Prescribed strategy; moving a handle to excite a whip does not establish these thresholds |
| Ordered dominant-bend stages | below 45%, then 45–75%, then above 75% cable length | Our engineered proxy |
| Local bend angle and persistence | ≥0.20/0.35/0.65 rad, each held ≥40 ms | Unvalidated numerical gates tied to a discretization |
| Feasible simulation/PVA execution | existing envelope and numerical checks | Keep separately; passing is not measured flight validation |

The wave stages must complete before the contact physics interval. If an
earlier contact fails a gate, later qualifying motion cannot produce success.
`failed` is a different boolean: it records modeled feasibility failures.
Both reviewed selected trajectories have `failed=false`, `success=false`.

The current preferred-fold reward is another separate construct. Its large
shape-reference term, timing alignment and weights are our design choices.
Passing hand-picked preference tests does not validate their necessity. The old
wave success flag is not directly rewarded or prioritized by the latest MPPI,
but still controls termination/cutoff. Changing it can therefore change rewards
and exported command length indirectly. Do not flip a flag and describe the
result as only a relabeling of the old experiment.

## Why the three-stage detector is not a reliable definition of a wave

For a smooth curve, local turning angle is approximately curvature times node
spacing. The audit samples the same 3-radian circular arc with 10 and 20 segments:
largest angles are 0.30 and 0.15 radians. A 0.20-radian gate switches from pass to
fail. Both actual old/new forecasts have the same mesh, so this is a validity
critique of the detector, not an explanation for their different shapes.

The detector sees only the largest angle and its location. It does not associate
one coherent fold between frames. Three independently held peak locations can
advance all three stages without supplying evidence of continuous transport.
Conversely, a real moving bend can lose the argmax to another bend, miss a band
between observations, or pass through it too quickly for the dwell requirement.
The latter can penalize faster propagation. Neither three completed stages nor
zero stages alone proves presence or absence of a physical travelling wave.

## Reinterpret the saved motions without rewriting them

| Saved motion | Modeled first tip contact | Speed/cone/reversal gates | Bend gate before contact | Original composite result |
|---|---|---|---|---|
| Sweep retained by timing trial | 1.124878 s | All pass: tip +X 4.179 m/s, 3D angle 34.46°, drone −1.298 m/s, retreat 17.52 cm | 0/3 | False |
| Flatter preferred-fold trial | 1.196966 s | Speed fails: tip +X 3.224 m/s; cone/reversal pass | 1/3 | False |

Both are geometric hits against our **5 cm virtual sphere**. That does not imply
success under the papers' smaller tolerances, a physical collision, or the
preferred forward-fold style. The retained sweep is rejected solely by the
added bend requirement among the listed contact/strategy gates. The flatter
motion fails both our speed and bend requirements.

## Original review recommendation

Define the target-contact outcome first, with the intended striking part and
actual target geometry explicit. A minimal proposed task result is geometric
contact by that part plus an executable trajectory. Tip-only versus distal-cable
contact remains a task-definition choice; the literature cannot choose it for us.
Retain numerical/command feasibility separately and keep physical validation as
a prospective measured check.

Report tip speed, direction, carrier motion and travelling-fold evidence as
separate measurements. Do not impose 4 m/s or three angle/dwell stages unless
an application requirement and validation justify them. For the user's desired
forward cast, examine whether a coherent fold moves toward the free end and
opens as the distal cable reaches outward. Keep that mechanism/style assessment
explicit; a disliked sweep must not silently become the desired whip just
because its tip touched the target.

Before any new optimization, agree on that observable outcome. Then a simple
distance objective with modest command effort is a defensible baseline to test;
do not automatically add a new detector, reference tracker or weight campaign.
This review recommends simplification but does not authorize or implement it.

Audit: `runs/audits/whip-success-review-20260910/condition_audit.json`,
`review.py`, and protected-file hashes. Analytic feature probes only; zero new
simulations. Original source, settings and frozen forecasts verified unchanged.

## Authorized implementation, 10 September

The user subsequently authorized changing the gate: the objective is to whip
and hit a target. MPPI now uses `task.success_criterion = tip_contact_v1`:
the tip enters the existing **5 cm radius virtual target sphere**, and the
trajectory passes the existing feasibility checks through the corresponding
30 Hz command boundary. Contact uses segment/sphere interpolation at the 150 Hz
physics clock. Speed, direction, reversal and the three-stage bend detector no
longer veto the hit. An earlier touch by another cable node also does not veto
a later tip hit. A non-tip touch alone is not success.

This changes the success definition, not the dynamics, target geometry, physical
bounds, reward weights, or preferred-fold objective. Those shaping terms still
influence search; they are not additional pass/fail requirements. A hit terminates
the planning episode, so earlier termination and contact-dependent reward terms
can change scores even though weights are unchanged. Future exports must still
check recovery separately. This is modeled contact, not physical validation or
proof of a coherent travelling fold.

Missing criterion versions retain `legacy_strike_v1`. Frozen jobs, original
forecasts and result flags were not rewritten. PPO's current configuration and
defaults are unchanged. The planner UI exposes the criterion and hides legacy
gate controls when geometric tip contact is selected.

Bounded fixed-command verification on Windows / RTX 4080, CUDA float64:

| Motion | Historical gate | New tip-contact gate | Contact time | Command cutoff |
|---|---|---|---|---|
| Retained sweep | False, feasible | True, feasible | 1.124878134 s | 34 |
| Flatter trial | False, feasible | True, feasible | 1.196966310 s | 36 |

Independent single-motion trace replays agree with the fused GPU batch contact
times within 1 ns; command prefixes agree exactly. A far-target miss remains
unsuccessful; an invalid starting height remains unsuccessful and infeasible.
The 37 focused tests passed, with three archived-fixture tests skipped. This
verification did not optimize, fit, fly, select a different rehearsal, or change
historical success labels. New-gate closest distances are truncated at contact
and should not be compared directly with old full-duration closest distances.

Implementation audit: `runs/audits/tip-contact-gate-20260910/verification.json`,
`verify.py`, original active MPPI configuration, and protected/source hashes.

## Subsequent authorized MPPI run

Run `20260910-022818-648386` completed with the new gate, using the previous
timing trial's exact settings/model/baselines except `success_criterion`.
512 random samples, full 1.5 s search, 20 updates to plateau, 64.475 s on the
RTX 4080. The best candidate is the previous sweep's exact first 34 commands:
modeled tip contact at 1.124878134 s, command cutoff 1.133333 s, feasible.
Objective 695.154 equals the best baseline under the new gate; no search
improvement or new preferred-fold behavior was obtained. About 1.60% of sampled
candidates hit and 71.19% were infeasible on average across updates.

Independent replay objective difference is 2.00e-10. The newly generated
rehearsal preserves this command prefix and checks the full recovery through
10.266667 s. Its 26 frozen files and 50 protected prior files were verified.
Rehearsal `20260910-022818-648386-M0-development-whip` is selected in Side XZ,
quarter speed, at 1.11 s. Search/supervisor finished; no fitting or flight.
Audit: `runs/audits/mppi-tip-contact-20260910/review.json` and `status.json`.
