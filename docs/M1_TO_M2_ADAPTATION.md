# M1-full to M2: reviewed flight adaptation

The user authorized another complete adaptation on 10 September 2026 and replaced
the proposed hard 4 m/s strike-speed minimum with a stronger soft speed reward.
The job is `runs/adaptation/M2-full-whip-v1`. Read its `status.json` before acting;
do not duplicate it. The parent is the exact flown **M1-full**, not the earlier
gain-only candidate also named M1. M2 is generation 2, parent M1-full.

## Measurements and split

All five new M1 flight pairs were first evaluated against their original frozen
M1 forecast. All 311 uploaded command rows match the selected CSV. The user
confirmed unchanged controller/hardware/CSV and no contact or intervention.
The flown body is cf_3, identified by its unique match to the controller's cached
measured XYZ; cf_6 does not match. This identification is based on measured streams.

New whole takes 001/002/004 train M2; 003/005 are excluded from optimization and
checkpoint selection. Their M1 outcomes were already inspected, so this is a
development holdout, not a blind test. Original M0 and preliminary roles remain
unchanged. No claim of physical M2 performance follows from this fit.

Coordinates remain raw global XYZ with the measured rotated attachment offset.
Clocks are estimated from measured streams, not aligned to a predicted trajectory.
Raw gaps and quality masks remain intact; no height normalization, coordinate
translation, prediction-based clock tuning or marker gap filling is introduced.
The entire 1.2 s M1 whip is scored. Initialization uses only 1 s past cable
observations and 0.4 s past drone observations. Actual timestamped commands drive
the recursive model; no measured state reset occurs inside the scored interval.
The frozen preparation exactly matches the earlier diagnostic arrays.

## Full model update

Fit all previously reviewed drone gains, effective delay and attitude response,
then the drone acceleration residual; refine attitude after the drone residual.
Fit cable EI, Cb and external damping with the inherited cable residual fixed,
then update that residual. Both networks start at their exact M1 weights. New
Adam state is appropriate because M2 has a different training objective and data.
The cable change regularizer compares to the inherited M1 output at the same
simulated states, rather than treating the parent residual as zero.

Training family weights are 50% new M1 flights, 25% earlier M0 adaptation flights,
and 25% preliminary training. Takes have equal weight within each family;
window count does not inflate a take's contribution. The earlier M0 replay uses
only 001/002/004, never 005 or reserved 003. The preliminary held-out figure8_002
remains excluded. There are 3 new whip, 3 prior whip and 69 preliminary drone
windows. Cable windows retain their separate causal initialization and masks.

The runtime is Windows, RTX 4080, CUDA float64, with batched candidates/windows,
captured execution and the existing full temporal cable derivatives. Neural
training has practical plateau stopping and no routine update ceiling. Current,
best, optimizer and stopping state are saved. Selected residual checkpoints must
pass full-rollout finite-difference checks. Nominal search guards, model bounds,
geometry, masses and residual acceleration bounds retain their reviewed meaning;
they are not measurements of motor or structural limits.

After selection freezes, compare M1-full and M2 on both new held-out takes with
identical observations, commands and masks. Also check the original preliminary
holdout and earlier M0 whip 005 for retention. Catalog status must summarize both
new held-out takes, rather than choosing the first one. Keep all per-take results.
Registration creates a candidate without automatic promotion or flight selection.

## Future M2 MPPI objective

`config/pva/m2_mppi.json` preserves the flown M1 task, search and bounds, and adds
the existing success-only outward tip-speed bonus with weight 400 and scale 4 m/s:

\[
R_{\mathrm{impact}}=400\frac{v_+^2}{16+v_+^2},\qquad
v_+=\max(\mathbf v_{\mathrm{tip,entry}}\cdot\mathbf d,0).
\]

It yields 80/200/276.9/320 points at 2/4/6/8 m/s. Four m/s is a smooth reward
scale, not a minimum or maximum. The bonus increases above 4 m/s. Misses and
infeasible trajectories receive none. Feasible tip contact remains the success
criterion and is prioritized before ranking successful candidates by objective.
No earlier-hit reward is added. This is a speed/energy proxy, not measured impact
force or power. A stronger bonus does not guarantee a faster or more accurate hit.

The previous hard-minimum requirement is preserved in the audit as superseded.
The flown M1 CSV, model, forecast, settings and historical success labels remain
unchanged. Global planner/PPO settings remain unchanged. The new settings are a
separate M2 recipe; no M2 planner or physical flight is launched by fitting.

## Completed result

The job completed in **1,667.3 s (27.8 min)** on Windows / RTX 4080. Drone residual
training stopped by plateau at update 260, cable physics at update 9, and cable
residual training at update 200 with best update 175 retained. Both selected NNs
passed full temporal numerical checks. Forty-two focused tests passed. The 70
protected raw/flight/model/config files remain unchanged. M2 is registered as a
candidate, **not promoted**. No M2 MPPI or real flight has run.

| New held-out takes 003/005, equal-take RMS | M1-full | M2 |
|---|---:|---:|
| Drone | 9.18 cm | 6.53 cm |
| Cable tip, measured drone input | 5.74 cm | 4.58 cm |
| Cable tip, command-driven prediction | 6.82 cm | 6.42 cm |

The combined result is mixed: 003 tip RMS improves 8.67→5.86 cm, but 005 worsens
4.97→6.99 cm. Across all five new takes, including training takes, drone RMS
improves 8.74→5.97 cm and conditional tip RMS improves 5.63→4.42 cm, while
command-driven tip RMS **worsens 5.85→6.23 cm**. Each component improves on every
new take individually; that does not guarantee the combined trajectory improves.
An exact decomposition of the saved predictions shows opposing cable and drone-
input errors in M1. Some of that cancellation changes after adaptation. This is
an algebraic diagnosis of saved trajectories, not a new physical force model or
proof that cancellation is the sole cause. No validation-driven retuning followed.

Earlier M0 holdout 005 improves: drone 7.64→5.78 cm, conditional tip 3.95→3.25 cm,
combined tip 8.81→8.20 cm. Original preliminary holdout mean-window error rises
slightly: drone 7.80→8.09 cm and conditional tip 4.92→5.04 cm. All retention checks
were performed after selection. The fitted effective delay is 0.08 s versus
M1's 0.06 s; estimated clock uncertainty prevents interpreting this as a measured
change in physical controller latency.

The candidate is `runs/adaptation/M2-full-whip-v1/candidate/model.json`, signature
`be4bd82c038dad1e2f1508babc9a1c3da121e1f428f2a375942b382bf6a7b0b9`.
The UI comparison is **Flight comparison → Compare generations → M2-full-whip-v1**.
Audit, per-take evidence and figures: `runs/audits/M2-full-adaptation-20260910`.
The next planning experiment can explicitly use this candidate and the separate
M2 soft-speed recipe. A successful modeled hit would still need prospective flight.
