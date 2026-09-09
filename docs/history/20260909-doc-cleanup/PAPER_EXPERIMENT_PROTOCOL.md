> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Paper experiments and figures

Design proposal, 2026-09-05. These are requirements for the next experiment
workflow, not claims that the current training runners implement all of them.
No learning curves or new policy-performance results have been generated.

## Fresh start

The user requested removal of previous exploratory checkpoints and results.
The cleanup removed 57 checkpoint/model files and cleared training runs,
validation previews, replays, audit outputs, temporary figures and fitting
search outputs. Old policy-result reports were removed or reduced to their
implementation descriptions. `runs/` starts empty.

Retained: preliminary raw recordings, synchronized processed recordings, the
derived force dataset, dataset roles, source code, configuration, literature,
and the physical baseline with its calibration provenance in
`data/cable_model/pivot_fit.json`. The baseline is a model calibrated before
policy deployment. It is not evidence of adaptation from failed policy trials.

Fresh paper training uses newly initialized PPO/SAC policies. Existing reward
and physical settings were not silently reset. Freeze and review the experiment
settings before launching the new paper runs.

## Figure plan

| Figure | Horizontal axis | Vertical axis / content | Interpretation |
| --- | --- | --- | --- |
| Baseline calibration | Time from window start [s] | Measured and simulated marker/tip positions; position error [cm] | Can the baseline predict unseen preliminary recordings? |
| Learning, main panel A | Simulated training attempts | Deterministic validation valid-hit rate [%] | Does the learned policy strike successfully? |
| Learning, main panel B | Simulated training attempts | Deterministic validation hit-and-recovery rate [%] | Does it complete the strike and return to settled hover? |
| Learning, secondary panel | Simulated training attempts | Mean deterministic evaluation return [reward units] | How does the shared task objective evolve? |
| Final policy outcomes | PPO / SAC on the same held-out scenarios | Per-seed success points, outcome counts and motion metrics | How do the frozen policies compare? |
| Representative strike | Time [s] | Tip distance, directed tip speed, velocity angle and point displacement | Why does a strike pass or fail? |
| Adaptation, future work | Number of real attempts used for adaptation | Held-out motion prediction error and real strike success | Does updating the model and policy improve transfer? |

For the baseline figure, use measured attachment motion to evaluate cable
parameters separately from force-tracking errors. Plot held-out takes, state
their number, and report marker and tip RMSE. Successive frames from the same
take are correlated and are not independent experimental repetitions.

The learning panels use the same x-axis, evaluation scenarios and legend. PPO
and SAC have different update rules, so optimizer-update counts are not a common
sample-budget axis. Also save actual simulator-step counts and wall-clock time;
report efficiency against these separately. Include nominal planning and
execution/recovery cost when reporting total simulation compute, and distinguish
validation cost from training cost.

Valid-hit rate counts valid tip-first contacts meeting the saved distance,
world-frame directed-speed and velocity-angle gates. Its denominator includes
every attempted validation trial, including refused plans and numerical failures.
Hit-and-recovery additionally requires settled PID recovery without failure.
Report plan refusal, invalid contact, non-tip-first contact, timeout and numerical
failure counts. Define mutually exclusive categories if presenting a stacked
outcome chart; the existing diagnostic flags can overlap.

Tip speed and angle at impact must identify their evaluated subset. A refused
plan has no impact measurement. Do not insert zero-valued impacts or silently
drop failed attempts from overall success rates.

The representative strike is selected by a rule fixed before viewing results,
for example the first trial of a saved scenario set. Show failures as well as
successes. Mark the predicted force-sequence cutoff and PID recovery on time
plots. Actual contacts are diagnostics and do not trigger early stopping.

## Repetitions and uncertainty

- Use independently initialized training seeds for each algorithm. My starting
  recommendation is five pilot seeds, aiming for ten for the final comparison
  when feasible. The necessary count depends on observed variability and the
  precision needed for the paper; five is not a universal sufficiency threshold.
- Publish all predeclared seeds, including poorly performing runs. Distinguish
  infrastructure failures from numerical/model failures and record restart rules.
- At each evaluation budget, draw the arithmetic mean across independent seeds
  with a pointwise 95% bootstrap confidence interval, resampling whole seeds.
  Optionally show faint individual-seed lines. Report the exact number of seeds.
- The curve interval is conditional on the fixed evaluation scenarios. It is
  not an interval over all possible real-world conditions. Final generalization
  evaluation needs a separate scenario/test set and an appropriate interval.
- Hundreds of validation episodes for one trained policy do not replace
  independently trained seeds. Repeated deterministic nominal trials also do not
  create independent training repetitions.
- Use shared evaluation budgets. Do not extrapolate stopped runs, hide missing
  seeds or change the number of contributing runs without labeling it.
- Display unsmoothed evaluation means and uncertainty in the paper. Optional
  dashboard smoothing must retain raw data and disclose its method and window.
  Never substitute a best-so-far curve for current-policy performance.

These choices follow the emphasis on reproducibility and uncertainty in
[Henderson et al., 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11694)
and [Agarwal et al., 2021](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).
The particular seed counts and figure arrangement above are project proposals.

## Common experiment contract

Both algorithms must use the same frozen physical baseline, task/reward
definition, action limits, initial-state distribution, execution protocol and
evaluation scenario files. Algorithm hyperparameters and compute budgets must
be recorded. Policy information access remains initial-state-only during plan
preparation, with no actual plant feedback controlling the strike sequence.
The measured initial state includes the drone attachment and all ten cable
markers, with estimated velocities; it is not limited to the drone state or an
assumption of a hanging cable. See the
[initial-state and open-loop contract](INITIAL_STATE_OPEN_LOOP.md) for the
verified node mapping and remaining live-estimation/training integration work.
Open-loop is the initial experiment; feedback-enabled maneuvers would be a
separately labeled later condition.

Decide the checkpoint-selection rule in advance and use it consistently across
algorithms. Learning curves evaluate the current saved policy; final results
identify whether they use the terminal or validation-selected checkpoint. Never
select the checkpoint using the final test set. Validation used by the PPO
update guard is part of model selection and cannot serve as untouched testing.

The present parameter perturbations are provisional simulation tolerances.
Label nominal and perturbed simulation performance separately. Do not describe
perturbed simulation success as real-hardware success or the tolerances as
measured hardware uncertainty.

## Data that must be saved for every new experiment

1. Run ID, algorithm, initialization seed, parent run if any, code revision and
   dirty-source hashes, dependency versions, hardware and full configuration.
2. Baseline and task IDs, immutable copies/hashes of the evaluation scenarios,
   data split IDs and evaluation seed.
3. An append-only training table: training attempts, valid planning transitions,
   nominal/plant physics steps, optimizer updates, elapsed time, task reward and
   algorithm-specific diagnostic losses. Keep entropy objectives separate from
   the shared task return used for comparison.
4. An append-only validation table plus per-trial outcomes: checkpoint identity,
   evaluation budget, hit, recovery, refusal, failures, reward, duration and
   motion metrics. Preserve every evaluation instead of overwriting its history.
5. The first validation trajectory for the dashboard, linked to that exact
   checkpoint, configuration and scenario. Its animation is an illustration;
   plotted success rates use the full validation set.
6. A final held-out evaluation record and the exact checkpoint it evaluates.

Implemented in the 2026-09-05 UI rework: append-only validation CSV/JSONL,
per-trial outcomes, scenario arrays and hashes, immutable evaluated checkpoints,
and the first actual validation trajectory. PPO/SAC UI launches request validation
after every collection batch. PPO explicitly labels reused evaluations when an
update is rejected. Run launch settings and baseline/task/configuration hashes
are saved. Learning and baseline fitting figures export PDF/SVG/600 dpi PNG with
numeric source data. Single-seed curves have no confidence interval.

Remaining work before final paper experiments includes seed-sweep orchestration,
cross-seed aggregation/uncertainty, complete compute/hardware accounting, and a
shared final-test selection/evaluation workflow. Existing algorithm-specific
selection rules have not been silently unified. Live initial-state estimation
and training coverage of measured cable shapes remain as documented in
INITIAL_STATE_OPEN_LOOP.md. Only temporary regression-test training has been run;
no new paper experiments have been started.

## Figure export

- Produce figures from saved numeric logs, using a script and committed plot
  settings. Export the plotted aggregate data and record source run IDs/hashes.
- Use white backgrounds, light grids, shared axis ranges and labels with units.
  Fix success axes to 0–100%. Prefer adjacent panels to dual-axis plots.
- Keep PPO blue and SAC orange, with distinguishable line styles as well as
  colors. The same algorithm uses the same style across every figure.
- Size for the selected journal's final column width; keep labels readable at
  that size (typically about 8–10 pt). Choose final dimensions after the venue
  is selected. Place legends where they do not cover data.
- Export vector PDF/SVG for line plots and a 600 dpi PNG for convenience.
  Export 3D trajectory panels as separate figures; screenshots of the whole
  application are not the quantitative figures.
- Captions state the metric definition, scenario set, number of training seeds,
  number of trials per evaluation, aggregation, confidence interval and any
  smoothing. Mark simulated and real-world measurements explicitly.

For the five-tab UI design, both PPO and SAC should show validation progress
curves alongside their own large 3D viewport. Keep losses and detailed reward
components in diagnostic logs. The Export figures control recreates learning figures from the same saved data, independent of window size.
