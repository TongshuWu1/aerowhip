# Paper-writing handoff

Updated 11 September 2026. The authoritative current sources are the
[whole-system paper audit](PAPER_PIPELINE_AUDIT.md) and the
[clean experiment protocol, release candidate 1](PAPER_EXPERIMENT_PROTOCOL.md).
The earlier version of this guide is preserved byte-for-byte in
`runs/audits/paper-pipeline-audit-20260911/before/docs/PAPER_WRITING_HANDOFF.md`.

## Research focus

The contribution is the complete UAV dynamic-whipping system: a reusable loaded
aircraft/cable model, offline MPPI-inspired planning, and measured real-to-sim-to-real
refinement between flights. Standard identification supports the system; a new
adaptation algorithm is not claimed. Commands are open loop with respect to cable
task feedback, while onboard UAV tracking feedback remains active.

Working title: **Targeted Aerial Whipping through Iterative Model Refinement and
Offline Sampling-Based Planning**. Final wording must match virtual interception
versus demonstrated physical contact.

## Method that actually exists

The implemented estimator is **staged, regularized nonlinear system identification
by simulation-error minimization**. See
[the frozen staged method](FROZEN_SYSTEM_IDENTIFICATION.md): nonlinear least squares
for aircraft response and cable parameters, Adam for both residuals, causal
initialization, training replay and selection frozen before validation. Combined
command-to-tip prediction is evaluated, not jointly optimized.

The model is a cascade from held 30 Hz desired PVA to effective loaded-UAV pose,
rotated cable attachment and rod dynamics. It has no explicit cable-reaction
feedback into the aircraft model. Residual bounds are discrepancy regularization,
not identified motor capability. The whole-maneuver search is MPPI-inspired
offline optimization, not exact path-integral control or real-time feedback MPC.

For the clean baseline, retain a fixed full model class from M0 through M2 and
the implemented staged method. Joint fitting remains an optional development
extension. A consistent fresh-M0 driver, execution/metrology checks and tested
launch readiness remain release gaps; do not describe them as completed.

## Current development evidence

Canonical flown lineage: **M0 → M1-full → M2-frozen-refit-v1**, with 5/5/3 real
takes. Gain-only M1 is a different sibling. The original M2 and its frozen refit
have the same model signature; they are not two updates. The selected global
flight is M2 run `20260910-211435-608306`, exactly as chosen by the user.

| Question | Current result | Qualification |
|---|---|---|
| Own original forecast tip RMS | 17.17 / 8.77 / 9.38 cm for M0/M1/M2 | Different plans and planner settings; not monotonic |
| Matched complete tip RMS on three fresh M2 takes | 14.86 / 7.99 / 7.03 cm | Identical diagnostic inputs; none of these models fitted those takes; M2 worsens take002 |
| Observed virtual entries | 0/5, 0/5, 0/3 at radius 5 cm | No current reliable real-hit demonstration |
| Frozen staged refit reproducibility | Same parameters, network tensors and reported losses/metrics | Same recorded environment/data; not a cross-device theorem |
| New launch audit | Drone start offsets up to 8.83 cm; estimated tip motion up to 0.136 m/s | Descriptive initial mismatch, not proof of the whole error cause |

All current runs and recordings are **development**, per the user's instruction.
They informed repeated method decisions and are not the future untouched paper
test. See [the detailed system comparison](M0_M1_M2_SYSTEM_COMPARISON.md) and the
new audit for scope, histories, masks, retention and numerical sensitivity.

## Manuscript structure

1. Explain the aerial free-tip target task and why model error matters during a
   short open-loop motion. Position it against dynamic rope manipulation and
   aerial flexible-cable control/throwing, using the audit's primary references.
2. Describe hardware, frames, packet execution, loaded response, rod mechanics,
   residuals and the staged identification objective with units and limitations.
3. Describe the exact offline planner, fixed reference/seed prior, target criterion,
   contact-speed preference, preparation and recovery.
4. Present the frozen clean experiment: roles, replay, model lineage, candidate
   adoption, matched prediction tests and blocked prospective real tests.
5. Report complete prediction error first, real targeting next, then component
   diagnostics, initial-state effects, compute/data cost and failures.

Recommended figures and their selection rules are in audit Section 10. Separate
original forecast, measured motion and initialized diagnostic prediction visually.
Do not call a plotted trajectory error the learned residual without explanation.
Use median-case examples for the main quantitative illustration; label any
best-case demonstration. Show missing data and individual take outcomes.

## Claims to keep conditional

Do not claim reliable physical hits, monotonic per-take improvement, independent
material-property identification, learned thrust saturation, measured impact power,
real-time MPPI, validated wave-detection gates, or broad hardware generalization
without the corresponding experiment. A tested system contribution does not
require proving a novel adaptation algorithm. Do not fill the clean study's
abstract/results with current development numbers.

The paper is ready to outline and write its verified methods now. Results and
strong performance claims wait for the released clean protocol and fresh data.
