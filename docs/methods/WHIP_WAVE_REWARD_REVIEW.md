# Preferred whip wave: evidence and reward-design review

Reviewed 10 September 2026 following the user's clarification that the latest
target-contact motion is not the desired whip. This is a literature and saved-motion
review, with two bounded diagnostic replays. No planner, reward, task, fit or
replay selection was changed; no optimization or flight was started.

## Main finding

The old preferred motion contains a substantially stronger travelling fold.
The current motion is not merely an equivalent wave rejected by an arbitrary
threshold. Our previous suggestion to remove the gate and prioritize contact
was incomplete: it could accept precisely the broad sweep the user dislikes.

The three-stage detector is still brittle. Both statements matter. Better
contact scoring and a better description of the desired motion are separate
requirements, and a changed success definition must never rewrite old results.

![Saved forecast comparison](../../runs/audits/preferred-whip-review-20260910/comparison.png)

In the middle row, the bright diagonal in the preferred motion is a strong bend
travelling toward the tip in material coordinates. The other two cases have
weaker, more diffuse curvature. Cyan points track the strongest bend and illustrate
why a single argmax can jump between features. The top row shows the actual fold.
Old X coordinates are translated only for plotting; original arrays are unchanged.

## Exact reference and comparison

The reference is the user's liked farther/lower-target result:
`20260909-173305-488091-mppi-wave`, held in the external archive. Its earlier
near-target outward-cast predecessor `20260909-172103-268459-mppi-wave` was also
checked and has similar folding. Both original forecasts remain in the archive,
with their original historical normalized M1 model identities. They are style
references, not accurate new-system predictions or new training measurements.

Current result: `20260910-011618-458510-M0-development-whip`. Both farther-target
cases have the same relative launch geometry and 0.9525 m cable rest length.
Metrics below use history through first geometric tip contact, not full recovery.

| Saved-motion feature | Preferred old forecast | Current forecast |
| --- | ---: | ---: |
| Largest local turning angle before contact | 0.711 rad | 0.259 rad |
| Peak sum of local turning angles | 3.054 rad | 1.600 rad |
| Peak tangent opposition, 0 parallel/no opposition to 1 opposite | 0.906 | 0.023 |
| Forward attachment-to-tip reach at contact | 0.713 m | 0.541 m |
| Tip velocity elevation at contact | +17.09 degrees | +34.27 degrees |
| Peak drone Z before contact | 2.105 m | 1.761 m |
| Three-stage detector before contact | 3/3 | 0/3 |

Turning angle is measured between adjacent unit tangents; curvature is angle
divided by the mean neighboring rest-segment length. Sum of turning angles
describes total bending, not mechanical energy. Tangent opposition is
`max(0, -min(dot(tangent_i,tangent_j)))`; it shows whether part of the cable folds
back against another part. It is invariant to rigid rotation and translation,
but a static folded cable can score highly, so it is not a wave detector alone.

The old near-target result gives 0.711 rad local turning, 2.979 rad total turning,
0.897 opposition and 0.717 m reach. Thus the qualitative contrast is not specific
to one old target. This remains a small, deliberately selected preference set.

## What the papers support

**McMillen and Goriely, Whip Waves (Physica D, 2003).** Their model studies a
travelling loop in a planar elastic rod, its interaction with taper, and unfolding
at the free end. It distinguishes motion of a shape feature from motion of a
material point. This supports examining a travelling fold, but not our particular
three regions or angle/dwell thresholds. Their tapered-rod crack analysis does
not establish supersonic behavior, an obligatory loop shape, or an energy reward
for our marker-loaded aerial cable. Author-hosted sections 4–5 were available via
indexed full-text passages; direct PDF fetch failed in this review.
[Author-hosted paper](https://goriely.com/wp-content/uploads/2003-PhysDwhip-waves-1.pdf).

**Krotov et al., Motor control beyond reach (Royal Society Open Science, 2022).**
The human study examines full whip kinematics, including successive marker-speed
peaks, extension and direction. Section 3.2 reports a generally proximal-to-distal
sequence with exceptions, including possible reflections; individual speed-profile
patterns do not visibly rank task performance. These are descriptive observations,
not an optimization reward or a mandatory success test. Their bullwhip and human
throwing setup differ from our drone and initially hanging cable.
[Paper, sections 3.2 and 4.2](https://pmc.ncbi.nlm.nih.gov/articles/PMC9533004/).

**Edraki et al., Human-Inspired Robot Whip Manipulation (ICRA workshop, 2025).**
This limited robot demonstration uses preparatory and striking minimum-jerk
motions with nine parameters. Equations 3–5 combine minimum tip distance with
effort; they do not enforce a three-stage curvature detector. The prescribed
motion family supplies structure that a distance objective alone does not.
The planar hinged whip and arm are different from our system; their preparation
direction is not an instruction to reverse our task's forward/backward convention.
[Paper, sections II-D and III](https://deformable-workshop.github.io/icra2025/spotlight/01_01_05_Edraki_Human.pdf).

**Chi et al., Iterative Residual Policy (RSS, 2022).** Section III defines rope
whipping through a three-parameter arm-motion primitive and minimum tip-to-goal
distance. Its iterative feedback improves actions across executions. This supports
structured motion and prospective correction, but their goal metric does not
distinguish our desired fold from another way of reaching the target. It cannot
justify calling every target contact the preferred wave style.
[Paper, sections III–IV](https://www.roboticsproceedings.org/rss18/p016.pdf).

The important distinction is between reproducing the papers' target-reaching
tasks and specifying the user's preferred travelling-fold task. None of these
papers validates our current numerical wave thresholds or provides ready-made
reward weights for this drone/cable system. The proposal below is our design.

## Tests that prevent the wrong conclusions

### Same old commands, new development model

One 39-command replay starts from X=0 and uses the exact archived accepted jerk
actions under the current development M0, with no optimization, time compression
or new limits. It is feasible, misses by 6.49 cm, and has peak total turning
1.491 rad, local turning 0.208 rad and tangent opposition 0. The old waveform
therefore does not survive simply reusing the old commands in the new model.

### Same old attachment motion, new cable model

A second diagnostic drives the new cable with the archived predicted attachment
trajectory, translated +2 m X, retaining the original initial hanging geometry.
This is a conditional boundary experiment, not a realizable drone command plan
or an original forecast. It recovers total turning 2.783 rad, local turning
0.487 rad and opposition 0.818, with 2/3 old detector stages. It does not contact
the target during the examined 1.3 s.

This shows that the new cable model can sustain substantial folding under the
old attachment motion. Predicted drone-motion differences, their timing, and
nonlinear cable response materially affect the weak fold in the same-command
test. Cable differences remain as well. It does not isolate a single bad fitted
parameter, prove either model physically correct, or justify reducing damping
until the preferred picture reappears.

The old and new models differ in mass, drone response, cable coefficients,
regularization and learned corrections. For example, nominal Cb is 2.5e-5 versus
about 1e-4, EI 1e-7 versus about 1e-8, and separate drag 0 versus 0.4/s. These
are provenance facts, not an ablation or a causal attribution to damping alone.

### Marker-speed cascade alone is insufficient

A simple adjacent peak-order check after peak forward drone speed ranks the
current motion at 90% strict adjacent ordering and the preferred motion at 70%.
Both have a progression of marker speeds. Our analysis is not a reproduction
of the human paper's time-normalized participant averages. Nevertheless, it
demonstrates that replacing the old detector with only this statistic could
favor the motion the user rejects. Use it as supporting evidence, not the main
style reward or a direct energy-transfer measurement.

## Why our last objective drifted toward a sweep

The whole-trajectory implementation made sample weighting and runtime more
practical, but changed task preferences too aggressively. It lowered maximum
wave credit from 120 to 15 and introduced up to 400 continuous joint-quality
points. The selected motion earns 266 joint-quality points and only 2.28 wave
points. It also removes the old horizontal-contact/near-target tip-direction
preferences from the actual optimization objective; legacy reward is merely
logged. The resulting closer, faster upward sweep is consistent with those
incentives. No single diagnostic proves the counterfactual optimizer outcome.

Another issue is timing: the new joint-quality score is maximized anywhere in
the 1.5 s trajectory, including after an invalid first contact. Here its peak
is at 1.1333 s, just after contact at 1.12488 s; approximate quality increases
from 256.15 at contact to 266.01 afterward. The difference is small here but
the semantics should be corrected before the next trial.

## Recommended reward design

Keep the faster full-trajectory search, same development M0 and unchanged physical
bounds. First construct a versioned objective offline and check its rankings on
saved motions. Do not start by trying another large set of numerical weights.

Use a small number of bounded terms:

`R = w_contact * Q_contact + w_wave * Q_fold_and_propagation
     + w_cast * Q_outward_cast - w_effort * C_command`.

1. **Contact quality:** distance, tip-first entry, forward speed and drone release
   evaluated at one consistent encounter time. Before contact, use continuous
   approach guidance. Freeze contact-related credit at first contact; do not
   assemble distance, reversal and velocity from different favorable times.

2. **Fold and propagation:** reward the formation of a coherent fold followed by
   its travel toward the free end and unfolding. Use rest-length curvature over
   the whole cable, with a soft reference to the preferred shape evolution in
   attachment-relative coordinates. Permit variable preparation/release timing
   with a monotone phase alignment. Do not prescribe the old drone path, copy
   its predictions into rollouts, or enforce a new height band. A softer
   spatiotemporal curvature-ridge measure can later replace the reference if it
   passes the same preference and negative-control checks.

3. **Outward cast:** near the encounter, reward attachment-to-tip forward reach
   together with forward tip speed, and softly penalize vertical/lateral tip
   velocity. Keep these preferences weak during preparation so the optimizer
   can fold the cable. The old 0.713 m reach and 17-degree elevation are
   descriptive style references, not newly measured bounds or mandatory targets.

4. **Effort and end state:** keep bounded, modest penalties for command effort,
   excessive carrier approach and difficult recovery. Do not maximize raw
   kinetic or bending energy: a big swing or stationary tight kink can game
   these, and cable motion is not purely an elastic spring-energy store.

For the first version, the preferred motion's shape history is a useful explicit
style reference: it captures what the user means better than an unvalidated
new one-number detector. Normalize position by cable length and remove attachment
translation, preserving strike-axis orientation. Timing alignment must cover the
whole preparation/fold/release sequence and be regularized; arbitrary time
warping must not let a static shape match a moving fold. This is a simulation
preference prior, not a physical demonstration or new fit dataset.

Do not treat a tiny wave term as optional decoration under a huge contact bonus.
Choose weights using fixed ranking examples so close, wave-like candidates rank
above comparable broad sweeps, while a beautiful stationary fold that never
approaches the target does not win. Report contact and wave quality separately;
preserve strict historical outcomes until a new criterion is explicitly adopted.

## Concrete next implementation check

Before another optimization, score the two preferred archived motions, current
contact motion, same-command new-M0 diagnostic, and fixed counterexamples:
rigid rotating straight cable, static folded cable, broad weak bend, reflected
wave, frame-shuffled shape sequence, and a fold that misses the target. Test
translation invariance, mesh sensitivity and timing variation. A total-turn
score alone fails the static-fold control; peak ordering alone already fails
the user's preference; a raw curvature maximum can reward a single-node kink.

Use the actual old accepted command pattern as one editable proposal, preserving
its pulse timing when represented by control points. It is distinct from the
earlier editable initial guess used to seed the recent runs. Optimize predicted
drone/cable behavior under development M0; identical command numbers are not
expected to reproduce identical attachment motion across models.

Only after the objective passes those checks should one MPPI trial be launched.
Continue the original sim-to-real-to-sim sequence: freeze forecast, record a real
whip, compare first, then use reviewed measurements to fit M1. The present review
does not authorize a new fit or establish real-world wave fidelity.

## Artifacts

`runs/audits/preferred-whip-review-20260910/` contains the comparison figure,
metrics, script snapshot, archive/source SHA-256 verification, a new fixed-command
prediction, a new conditional attachment prediction, model-parameter comparison
and speed-cascade diagnostic. The 18 inspected original artifact files remain
unchanged. No old archive is restored into active model or rehearsal selection.
