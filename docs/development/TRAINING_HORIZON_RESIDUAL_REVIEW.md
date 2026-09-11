# Training windows, residual evidence and stopping

Reviewed against source, measured-data artifacts and primary papers on 10 September
2026. This review accompanies the complete adaptation, not a change to the flown
commands or original M0 forecast. Current continuation: `M1-full-whip-v2`.

## What the implementation actually uses

| Quantity | Duration | Purpose |
|---|---:|---|
| Past drone history | 0.4 s | Estimate starting velocity, effective compensation and attitude alignment |
| Preliminary drone prediction window | 2 s | Fit recursive command-to-pose response over varied motions |
| Measured whip prediction window | 1.1333 s | Include the complete commanded whip segment |
| Past cable history | 1 s | Estimate initial cable shape/velocity with existing endpoint weighting |
| Preliminary cable prediction window | 1 s | Fit cable dynamics under measured attachment motion |
| MPPI lookahead for the selected flight | 1.5 s | Planning horizon; independent of fitting/history windows |

These are different quantities. The 1.1333 s maneuver is followed by recovery in
the 10.2667 s command CSV; the fit does not claim to train on that entire recovery.
No measured samples inside a prediction window are used to reset the predicted
drone. Conditional cable fitting uses measured attachment as its boundary; final
combined predictions use the predicted attachment instead.

The two-second drone duration is a project choice inherited from preliminary
identification, not a paper-established optimum. It supplies more than the short
whip duration and exposes position/velocity response over varying preliminary
commands. One-second cable windows also cover substantially more than a single
integration step. Neither duration has been proven optimal on this system.

## Does longer training mean better prediction?

Not necessarily. A longer rollout penalizes accumulated model drift, but also
amplifies initial-state error, command-clock uncertainty and numerical sensitivity.
It costs more per gradient evaluation and creates fewer separately initialized
windows from a fixed recording. Longer recordings containing diverse motion are
valuable; that is different from increasing every recursive training window.

DEFORM uses one-second, 100-step training rollouts and evaluates predictions out
to five seconds. Its ablation supports multi-step over one-step training, not an
unbounded benefit from ever-longer windows. It separately tests residual learning
and physical parameter learning. Our free-tip aerial cable and acceleration MLP
are not its two-manipulator boundary and integration-correction GNN.
[Chen et al., Sections 5.1–5.2 and Appendix B.4](https://arxiv.org/html/2406.05931v2)

Ribeiro et al. show why long recursive simulation can make noncontractive model
identification ill-conditioned, and analyze multiple shooting. Their equivalent
multiple-shooting formulation uses continuity constraints; our independently
initialized training windows are not that exact constrained formulation.
[Ribeiro et al.](https://arxiv.org/abs/1905.00820)

Mamedov et al. train on one-second rollouts from a single trajectory, then test
other motions and longer predictions. They distinguish training initial-state
optimization from past-data state estimation at test time, and report the
importance of exciting training motion. This supports explicitly auditing initial
state and data coverage, rather than treating duration as the only design choice.
[Mamedov et al., Sections 4.3 and 5](https://arxiv.org/html/2407.03476v1)

The appropriate next measurement is error versus elapsed prediction time at
matched starts, with fixed commands, histories and masks. Such curves diagnose
drift; they do not, by themselves, prove that training with a longer window helps.
A controlled training-horizon comparison would require separately trained models,
the same data split and optimization opportunity. Do not tune this repeatedly on
005 and call it an untouched test. Keep current training durations while resolving
the demonstrated numerical issue and completing the full candidate.

## Are both residuals necessary?

They address different discrepancies. The drone residual changes command-to-pose
prediction. The cable residual changes motion under the attachment boundary.
Neither network is automatically necessary, and training loss alone cannot prove
either improves new flights.

Torrente et al. learn nominal quadrotor acceleration discrepancies and compare
nominal, linear-drag and GP-augmented models on common trajectories and data. That
is evidence for evaluating residual contributions, not proof that our two neural
networks are needed. Their motor-thrust model/interface differs from our empirical
closed-loop PVA response.
[Torrente et al., Sections III-F and IV](https://rpg.ifi.uzh.ch/docs/RAL21_Torrente.pdf)

The current drone already inherited a nonzero M0 residual. Comparing the saved
nominal stage against the adapted-drone stage measures the benefit of **updating**
that residual. It is not a comparison against a freshly fitted model with no NN.
Turning a residual off with co-adapted nominal parameters held fixed is a useful
removal diagnostic, but can overstate necessity compared with refitting a strong
non-neural alternative. These claims must be kept separate.

The final check should retain the intermediate stages and compare:

1. M0 and updated nominal response with the inherited drone NN.
2. The same updated response with the adapted drone NN.
3. Adapted cable physics without a cable NN, under measured attachment motion.
4. The same physics with the cable NN, then all blocks together under commands.

Also evaluate all four combinations of drone/cable residual enabled/disabled at
fixed final nominal parameters as a clearly labeled removal diagnostic. Report
drone position/attitude, conditional cable tip and command-driven tip separately,
per take, with preliminary-data retention. Select/freeze weights before these
comparisons. No automatic model or flight promotion follows from the results.

## What the current run has established

The first full attempt, `M1-full-whip-v1`, completed nominal drone fitting, 400
drone NN updates, attitude refinement and a three-parameter cable fit. Drone NN
training objective decreased from 4.810872 to 3.884971 (19.25%). This is training
evidence, not held-out improvement. The original M0 is unchanged.

The cable NN made **zero updates** in that attempt: its full-window gradient check
failed at the final physical iterate. Dense and optimized solver comparisons
localized the sensitivity to preliminary window `osci_002-01641-0`. Reducing the
integration step changes that window's predicted motion and does not yet show
clean step convergence; no substep or damping-regularization change was adopted.
All eligible windows remain in the dataset.

The preceding physical iterate improves the same training objective from
1.928873 to 1.886859 and passes the full NN gradient check. Its coefficients are
EI 9.9972936e-9, Cb 9.9145410e-5 and external damping 0.3660089/s. The first and
final iterates, failed checks and resolution study are preserved. The continuation
uses the best numerically verified physical iterate, not the lowest raw loss
regardless of numerical reliability. Selection uses training data only.

## Why the ceiling changed

The 400-update ceiling interrupted the drone NN while it was still making useful
progress. It was an engineering guard, not a scientific stopping criterion.
The authorized continuation removes the routine neural update ceiling and
preserves the Adam state, best weights and original loss reference. Reconstructing
the saved update-400 objective reproduced 3.884970568162815 exactly.

Stopping now uses the declared practical-progress rule: minimum 40 updates,
evaluation every five updates, and six checks without a cumulative 0.5% meaningful
improvement. Best weights are retained even when smaller improvements occur.
Manual stop and finite-value/gradient checks remain. This rule can stop a run
without claiming global convergence; it avoids both an arbitrary low ceiling and
unbounded training for negligible gains. The 80-evaluation guard on small nominal
least-squares searches is separate; those stages terminated before it.

Both residual stages preserve successive best checkpoints. Final selection must
also pass full-rollout numerical verification. A failed numerical checkpoint is
retained as evidence, not selected using held-out accuracy. This does not prove
all derivatives or future states are well-conditioned.

See `runs/audits/full-adaptation-20260910` for the resolution tests, frozen records,
resume check and resulting model comparisons. Check current job status before
interpreting any pending stage as completed.

The continuation is now **completed**. The cable NN stopped at update 280 with
training objective 1.147372, down from 1.886620 (39.18%). Its selected-checkpoint
derivative agrees with the central difference within 0.282% at the checked step,
with the adjacent step also passing. The full candidate and both component
checkpoints were frozen before their combined validation.

## Frozen drone-stage checks

The continued drone residual reached practical plateau at update 585 and passed
its numerical check. Its training objective is 3.743476, versus 3.884971 at the
old update-400 ceiling. The original objective and Adam state were preserved.

The following position metrics use the same native-pose samples for each model;
they are not the slightly different 150 Hz grid metrics used in the combined
drone/cable comparison. All units are centimetres.

| Frozen stage | Whip 005 position RMS | Preliminary holdout mean 2 s RMS | Matched-start 4 s RMS |
|---|---:|---:|---:|
| M0, inherited residual | 8.9193 | 9.8075 | 12.4638 |
| Updated nominal, inherited residual | 7.7626 | 10.3282 | 12.5475 |
| Updated nominal and drone residual | 7.6321 | 7.7974 | 10.0701 |

This is evidence that updating the drone residual helps prediction on these
excluded recordings, especially the preliminary holdout. Its incremental benefit
on whip 005 is small. Relative to the nominal-update stage, 005's X/Y/Z RMS changes
from 4.272/2.679/5.902 cm to 6.024/1.307/4.501 cm: X worsens while Y and Z improve.
Attitude RMS changes from 0.06818 to 0.07466 rad after the post-NN refinement;
both improve on M0's 0.10744 rad. Do not describe every component as improved.

The 4 s check uses 25 matched starts from the original preliminary holdout. It
excludes one start without complete pose/command coverage, and its windows
overlap. The models were trained on the original 2 s windows; no 4 s model was
trained. These are descriptive prediction-duration checks, not independent
replicates or a training-horizon ablation. Data and model hashes, full per-take
results and the figure are in `runs/audits/full-adaptation-20260910/drone-stages`.

## Actual residual-removal results

All rows below keep the final nominal drone and cable parameters fixed. They use
the same initialization, commanded packets, observations and scored interval on
whip 005. Units are centimetres, on the combined comparison's 150 Hz grid.

| Residual enabled | Drone RMS | Conditional cable-tip RMS | Command-driven tip RMS |
|---|---:|---:|---:|
| Neither | 6.2633 | 6.5799 | 10.1001 |
| Drone only | 7.6352 | 6.5799 | 11.5659 |
| Cable only | 6.2633 | 3.9527 | 8.2092 |
| Both | 7.6352 | 3.9527 | 8.8111 |

The cable residual helps this whip under both boundary conditions. The drone
residual is not helpful on 005 relative to removing it. However, removing it
worsens the original preliminary holdout's mean drone RMS from 7.7974 to
12.7054 cm. It also worsens drone position on all three adaptation whip takes.
Updating the inherited residual, removing it and refitting a strong non-neural
alternative answer different questions; only the first two were tested here.

The cable NN also has a retention tradeoff: preliminary holdout conditional tip
is 4.8207 cm with updated physics alone and 4.9231 cm with the cable NN. M0 is
4.9401 cm. Do not describe this as a meaningful improvement on every motion.
Take 004's command-driven tip also worsens from 10.7077 cm without the cable NN
to 10.9816 cm with it, despite improved conditional cable prediction.

My current recommendation is to retain both residuals in the frozen full
candidate because they provide complementary benefits across the checked
motions, while keeping the above tradeoffs explicit. This is not proof that
both networks are necessary. Do not select the cable-only diagnostic because it
wins on the already inspected 005, then call that independent model validation.
The original M0 remains selected; a later prospective flight is still required.

Artifacts: `runs/evaluation/M1-full-whip-components`,
`runs/audits/full-adaptation-20260910/drone-residual-off`,
`preliminary_cable_stage.json`, and `components/residual_removal.png` in the same audit.

## Review of hard limits

The user clarified that all limits should be judged by effectiveness, not removed
blindly. No arbitrary command-acceleration cap was added. The current findings are:

| Setting | Evidence | Decision for this frozen candidate |
|---|---|---|
| Neural update ceiling | 400 interrupted useful drone progress; continuation stopped at 585/280 by plateau | Removed from this continuation; preserve historical protocol settings |
| Nominal 80-evaluation guard | Each of nine drone searches used 11 evaluations; cable used 8, final attitude 7 | Not binding; increasing it would not change these stopping decisions |
| Six drone-gain and three attitude-parameter bounds | Fitted values are interior; all returned active-bound flags zero | Keep; no evidence these limits prevent a better fit |
| Effective delay grid, 0–120 ms | Best declared delay 60 ms, interior; adjacent grid scores higher | No range expansion justified; resolution optimality and actuator latency are not established |
| EI/Cb/external damping bounds | Selected values interior; numerical reliability, not a bound, rejected the final physical iterate | Keep; removing bounds does not resolve the demonstrated sensitivity |
| Drone NN ±0.5 m/s² per axis | 6.08% of sampled training components exceed 95% of this bound | Sometimes active; a tunable discrepancy assumption, not a vehicle limit |
| Cable NN ±0.5 m/s² per free-node axis | 6.34% of sampled training components exceed 95% of this bound | Sometimes active; current network improves whip prediction without changing it |
| Optimizer gradient-norm clip 1.0 | Large gradient spikes occur during cable training | Keep for numerical stability; this does not clip predicted or commanded acceleration |
| Finite-value, geometry, source and observation-validity checks | Guard against invalid predictions and missing/changed evidence | Keep; do not relax these to make a candidate appear successful |

The residual magnitude check uses training inputs only. Drone RMS corrections per
axis are 0.256/0.200/0.236 m/s²; cable values are 0.301/0.104/0.294 m/s². Fractions
are descriptive, unweighted counts of sampled components, not independent data.
Cable corrections were sampled at outer 150 Hz states, not every integrator
substep. Both bounds are occasionally active, so they have **not** been proven
nonrestrictive. However, the observed validation tradeoffs do not justify blindly
increasing correction freedom. A wider-bound claim needs its own controlled fit
and excluded-data comparison; no such fit was run or silently substituted here.
The ±0.5 values are project choices, not recommendations taken from the papers.

All original M0/MPPI/flight assets and prior model entries were verified unchanged.
The final UI and 38 targeted tests passed on native Windows/RTX 4080. M1-full is
registered for review; registration does not select a flight model. Full results
and provenance are in `docs/methods/FULL_MODEL_ADAPTATION.md` and the associated audit.
