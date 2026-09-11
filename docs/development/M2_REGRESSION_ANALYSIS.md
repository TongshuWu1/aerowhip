# Why M2 improves the components but not every complete prediction

Analysis completed 10 September 2026 for `M2-full-whip-v1`, whose parent is the
flown **M1-full**. The user requested an explanation, not another optimization.
All analysis uses frozen models and the same reviewed M1 flight observations,
commands, causal initial states and masks. M2 remains unpromoted. No fitting,
MPPI, flight, controller change or historical-forecast replacement occurred.

**The main finding is a gap between the fitting objective and the prediction
needed by MPPI.** The drone fit rewards average tracked-origin position accuracy.
The cable fit rewards cable accuracy when driven by the *measured* attachment
motion. Neither fitting objective evaluates the complete command-to-tip chain.
That chain is evaluated only after the selected parameters are frozen. M2 learns
a larger forward excursion that improves average drone position, but drives the
cable into an excessive upward arc on three of the five repetitions. Controlled
component and coordinate swaps reproduce this effect. Changed cancellation of
the old model's errors also contributes; cancellation alone is not the explanation.

**Observed performance.** Values are equal-take means of Euclidean RMS over the
scored 1.2 s whip, not pooled frame errors or target-hit probabilities. These are
common causal-history diagnostic predictions, not replacements for the original
forecast saved before the real M1 flights.

| All five new takes, including adaptation takes | M1-full | M2 |
|---|---:|---:|
| Drone position RMS | 8.74 cm | 5.97 cm |
| Tip RMS with measured drone input | 5.63 cm | 4.42 cm |
| Tip RMS with predicted drone input | 5.85 cm | 6.23 cm |

| Take | Role | Complete tip RMS, M1-full → M2 |
|---|---|---:|
| 001 | Adaptation | 4.64 → 6.66 cm |
| 002 | Adaptation | 6.05 → 5.60 cm |
| 003 | Development holdout | 8.67 → 5.86 cm |
| 004 | Adaptation | 4.90 → 6.07 cm |
| 005 | Development holdout | 4.97 → 6.99 cm |

Drone position and measured-input cable prediction improve on every new take.
Complete prediction worsens on 001, 004 and 005. The adaptation-take mean itself
worsens, 5.20 → 6.11 cm, so this cannot be explained solely as held-out overfitting.
The two holdouts average 6.82 → 6.42 cm because the large improvement on 003 exceeds
the regression on 005. That average does not establish consistent improvement.

**What the fitting code actually optimizes.**

- `experimental_data/whip_full_optim.py`, `FullTranslation.rollout`: recursively
  integrates the commanded drone response, then applies robust position loss to
  the tracked origin. Nominal priors and residual magnitude/change penalties are
  additional terms. There is no supervised attachment-motion, velocity or combined
  tip loss. Attitude is fitted separately after the translation update.
- `experimental_data/whip_full_data.py`, `cable_rows`: assigns `truth[:,0]` to the
  cable's driving boundary, using measured origin plus the measured rotated
  attachment offset.
- `experimental_data/whip_full_cable.py`: fits physical parameters and then the
  residual using recursive cable rollouts with that measured boundary. The cable
  state is not reset to observations during the window. The data term combines
  all-marker and tip robust position errors.
- `experimental_data/whip_full_fit.py`: freezes selection before
  `combined_validation`. There is no final coupled fitting stage or training-time
  command-to-tip checkpoint criterion.

Thus all requested model blocks were updated, but they were optimized separately.
Calling that a joint fit of the complete prediction would be inaccurate. A
decrease in both separate losses does not mathematically require a decrease in
the composed prediction error.

**Controlled component swaps.** Two frozen cable models were each driven by the
saved M1 and M2 predicted attachment histories, with identical initial cable
states. The diagonal cases reproduce the existing complete predictions to within
1.83e-12 m on Windows/RTX 4080. Crossed cases are diagnostic boundary histories,
not new candidate models or flyable commands.

| Mean complete tip RMS across all five takes | M1 cable | M2 cable |
|---|---:|---:|
| M1 predicted drone attachment | 5.85 cm | 6.33 cm |
| M2 predicted drone attachment | 6.44 cm | 6.23 cm |
| Measured attachment, conditional reference | 5.63 cm | 4.42 cm |

The M2 cable is better with measured input but worse when substituted under the
old predicted input. The M2 drone input is worse under the old cable on average.
Neither component substitution alone fixes the combined result. The diagonal M2
combination partially offsets these effects, but not enough to beat M1 on average.

**The motion responsible for the regression.** An additional fixed factorial
diagnostic replaces individual XYZ coordinates of the saved M1 attachment with
M2 coordinates, while retaining the M2 cable. It does not optimize a parameter.

| M2 cable, tip RMS | 001 | 003 | 005 |
|---|---:|---:|---:|
| All attachment coordinates from M1 | 5.43 cm | 8.65 cm | 5.88 cm |
| Only attachment X changed to M2 | 6.50 cm | 6.35 cm | 6.84 cm |
| All attachment coordinates from M2 | 6.66 cm | 5.86 cm | 6.99 cm |

The X-history change alone reproduces most of the regression on 001/005 and much
of the improvement on 003. This is a forward-motion effect producing a tip-height
error through the cable dynamics; it is not simply the drone being predicted too
high. Changing only Z to M2 actually improves these three cases relative to the
all-M1 attachment history under the same M2 cable. Coordinate effects interact,
so the table is not an additive partition of causality.

At approximately 0.75 s on take 005, measured attachment X is 0.585 m, M1 predicts
0.602 m and M2 predicts 0.643 m. On 003, measured X is 0.691 m, M1 predicts 0.625 m
and M2 predicts 0.666 m. The same learned increase helps the larger excursion and
overshoots the smaller excursion during this phase, while reducing later X error
on both takes. Average position RMS hides that time-dependent tradeoff.

The measured maximum tip heights are 1.213 m on 001 and 1.173 m on 005; M2 predicts
1.300 m and 1.265 m. In contrast, measured 003 reaches 1.302 m and M2 predicts
1.294 m. On the three regressing takes, the tip RMS in the final 0.9–1.2 s interval
changes as follows: 001 4.57 → 10.56 cm; 004 4.86 → 9.32 cm; 005 5.98 → 11.08 cm.

![Take 005 attachment motion and tip height](../../runs/audits/M2-regression-analysis-20260910/why_tip_rises.png)

Attachment velocity estimates also show why position RMS is incomplete. Using
the same 80 ms local polynomial derivative on measurements and predictions,
velocity RMS changes 0.377 → 0.393 m/s on 001, 0.347 → 0.376 on 004 and
0.357 → 0.387 on 005. Both tested smoothing spans, 40 and 80 ms, give the same
direction. These are noisy derivative estimates, not directly measured velocity;
they do not prove that adding a velocity term alone will fix the model.

**Old errors sometimes cancel.** Let C be a frozen cable simulator, r the measured
attachment, r-hat the predicted attachment, and y the measured tip. With the same
initial cable state,

\[
e_{\rm total}=C(\hat r)-y
=\underbrace{C(r)-y}_{e_{\rm conditional}}
+\underbrace{C(\hat r)-C(r)}_{\Delta_{\rm input}}.
\]

Therefore total MSE equals conditional MSE plus input-effect MSE plus
`2 mean(dot(e_conditional, delta_input))`. This is an exact algebraic identity,
without a linearity assumption. The cross term is negative on all five takes.
For 005, in cm², M1 has `34.52 + 60.50 - 70.31 = 24.71`; M2 has
`22.57 + 85.67 - 59.43 = 48.82`. The standalone cable gets better, while the
effect of its imperfect driving motion grows and cancellation weakens. For 004,
cancellation becomes stronger but input-effect MSE grows enough to cause a
regression anyway. It would be incorrect to attribute every regression solely
to lost cancellation, or to describe M1's smaller combined error as proof that
its individual physics were more accurate.

**Secondary issues and uncertainty.** Attitude RMS worsens on every new take,
roughly 4.6–5.9° → 5.6–6.6°. Replacing only M2's rotated attachment offset with
M1's reduces the combined mean from 6.23 to 6.04 cm; most of the problem remains.
The main measured intervention effect comes from translation.

The effective response delay changes 60 → 80 ms. The 60/80 ms nominal profile
scores are 4.16779/4.15605, only about 0.28% apart. This is weak evidence for a
unique physical latency, especially with estimated recording clocks and about
32 ms chunk spread. It is not evidence that the actual controller delay changed.
A separate fixed-prediction sensitivity check shifts measured tip timing by
-40/-20/0/+20/+40 ms on a common interior mask. The regression sign on 001/004/005
and improvement sign on 003 persist throughout. The small mean difference remains
sensitive in magnitude, and 002 changes sign at +40 ms. This check does not
reinitialize or reintegrate under shifted clocks, so it is only a partial clock
uncertainty test; no offset is selected.

Repeated commands produce materially different measured excursions. The current
input/state representation predicts much less of that variation. These records
do not identify whether its physical source is an unobserved operating condition,
imperfect initial-state estimation or missing dynamics. Do not label it a battery
problem or a newly measured acceleration limit. Three training repetitions and
two previously inspected development holdouts cannot establish broad physical
generalization or separate every nominal and residual parameter uniquely.

There is no evidence here that the new impact reward caused the regression: it
has not been used by a new planner and is absent from fitting. Existing selected
residual gradient checks passed, and the new frozen-driver rollouts reproduce
production arrays. This makes a replay/composition arithmetic mismatch an unlikely
explanation for these centimetre-scale effects; it does not prove the physical
model is exact or rule out every implementation issue. No bound was changed or
shown to be the cause by these diagnostics.

**Literature context and proposed next step.**

[Mamedov et al., Learning deformable linear object dynamics from a single
trajectory](https://arxiv.org/html/2407.03476v1) model DLO motion with manipulated-end
pose, velocity and acceleration as inputs, and endpoint position and velocity as
outputs (§3). Their rollout objective fits those outputs with regularization
(§4.4). Their measured/kinematically computed boundary inputs differ from our
additional requirement to predict that boundary from drone commands. Their
Appendix C uses staged initialization followed by full parameter optimization;
it does not establish that joint drone/cable refinement will work for our setup.
Their filtering settings were chosen for their data and should not be copied
unchanged to the faster whip. The following proposal is an inference for this
project, not a paper-prescribed recipe.

Keep the component stages as initialization and diagnostics. Add a modest final
refinement whose loss includes the complete recursive command-to-marker/tip
prediction, alongside drone pose and measured-input cable terms and the existing
parameter/residual regularization. In schematic form,

\[
L=\lambda_d L_{\rm drone}
+\lambda_c L_{\rm cable\mid measured\ root}
+\lambda_f L_{\rm cable\mid predicted\ root}
+\lambda_r L_{\rm regularization}.
\]

The component terms are necessary: optimizing only the final tip could deliberately
make the cable compensate for a wrong drone. Attachment trajectory/attitude and
carefully estimated velocity should be monitored explicitly; adopting a derivative
loss requires checking its noise and timing sensitivity first. Do not simply
increase epochs, remove bounds, force an old X trajectory, or change the strike
reward to conceal prediction error.

Use adaptation takes and prior training replay for fitting and checkpoint choice.
Before running, freeze the loss scaling, stopping and retention criteria. Include
the complete prediction in the training criterion and predeclare reporting by
take, axis and phase. Preserve both current holdouts and old retention sets; since
their outcomes now inform the method, any successor remains a development result.
A later prospective flight is needed to test the changed method independently.
This final refinement is **proposed, not implemented or run by this analysis**.

Evidence: [audit and reproducible scripts](../../runs/audits/M2-regression-analysis-20260910/README.md),
[original M2 fit report](M1_TO_M2_ADAPTATION.md), and
[exact saved-error decomposition](../../runs/audits/M2-full-adaptation-20260910/combined_error.json).
