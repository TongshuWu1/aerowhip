# Comparing M0 to M1 to M2

Read [the current system comparison](../development/M0_M1_M2_SYSTEM_COMPARISON.md) for exact
model identities, all 13 flights, matched-model results, related work and the
clean paper design. All current runs are development evidence.

Open **Flight comparison → Study overview** for three flown generations. Choose
**Real flights · original forecasts** for the deployed predictions, or
**Same commands · compare models** for a common causal initialization and observed
command sequence. The default matched-model filter excludes any take used by any
compared model, including ancestor training replay. M2 takes 001/002 have future
adaptation roles but were never used to fit M0/M1/M2.

M0, M1-full and M2-frozen-refit-v1 have 5, 5 and 3 real takes. The rejected
gain-only M1 and duplicate M2 reproducibility refit remain accessible under
**Compare generations**. They are variants, not additional successive adaptations.
The selected flight CSV and every original ghost are preserved. The latest model
comparison does not change any flight selection.

Use a table row to replay its actual flight with the original flown ghost.
A diagnostic prediction under a different model never replaces that ghost.
Select **Model variants, traces & fitting** for the existing detailed tools.
Empty or changed evidence clears the overview; viewing does not fit, plan or
evaluate. **Detailed review** opens the paper-oriented explanation and limitations.

The evidence pointer is `config/evaluation/system_review.json`; it binds an
immutable report and source hashes. Current planner preferences are not scientific
inputs to an older frozen comparison. Updating a future study requires explicitly
creating and registering a new reviewed report.

Previous chronological UI/status notes are preserved in
`runs/evaluation/M0-M1-M2-system-review-20260910-v2/docs_before/docs/methods/SIM_REAL_EVALUATION.md`.
The original evaluation design below retains its general workflow; any old
statements that no M1/M2 flights exist are superseded by the current review.

## The questions we will answer

1. **Did prediction improve?** Run every registered model on the same reviewed
   recorded commands and initial state. Compare errors on identical observations.
2. **Did the real task improve?** Fly a newly frozen plan and compare the recording
   with that plan's original forecast. Report target outcome and execution too.
3. **What did improvement cost?** Retain the number of real takes, usable recorded
   seconds, fit updates/time and planner samples/time at each round.

The first comparison isolates a model change under fixed inputs. The second
measures the combined model-and-planner experiment when the command changes.
A lower fitting loss answers neither question by itself. Replaying an old CSV
under M1 produces a new diagnostic; it never replaces M0's original prediction.

## What related work actually evaluates

| Primary source checked | Evaluation principle | Our application |
| --- | --- | --- |
| [SimOpt, Sections III–IV](https://arxiv.org/pdf/1810.05687) | Alternates simulation adaptation and policy learning; examines transfer success and the number of real trials/iterations. Its simulator update compares simulated and real observations under the current policy. | Preserve each round and its real-data cost; distinguish identification from the downstream planning result. We use MPPI and a scalar physical update, not SimOpt's learned parameter distribution. |
| [COMPASS, Sections 4.2–4.3 and Table 1](https://proceedings.mlr.press/v229/huang23c/huang23c.pdf) | Reports trajectory discrepancy separately from goal distance and real task success, with repeated independent policy runs. Parameter values need not equal ground truth to reduce discrepancy. | Report prediction error and task success separately. Effective damping is not a measured aerodynamic coefficient. A single model sequence is a case study, not multiple independent learning seeds. |
| [Mamedov et al., Sections 4.3, 5.1 and 5.3](https://arxiv.org/html/2407.03476v1) | Tests different trajectories, frequencies and amplitudes, reports endpoint prediction and runtime, and studies longer horizons. Initial-state estimation is explicit. | Use past-only initialization and report errors across time. Repeating one command tests repeatability; a different motion is needed to test motion generalization. |
| [DEFORM, Section 5](https://arxiv.org/html/2406.05931v2) | Uses motion-capture observations, compares recursive multistep prediction with baselines, and evaluates runtime and component ablations. | Score recursive drone/cable prediction rather than just one-step fit loss. Add residuals or ablations only when a concrete diagnostic justifies them. |

The cited methods/results sections were read directly. These principles motivate
this protocol; the papers do not establish our numerical thresholds, data budget
or target radius. Their hardware and manipulation tasks differ from aerial whipping.

## Minimal experiment for this system

**Round 0:** preserve the selected M0, its exact 30 Hz PVA CSV, original forecast,
planner settings and source hashes. Execute repeat takes with the same documented
hardware/controller. Whip 001/002 are predeclared adaptation; 003 is validation.
This is a practical pilot split, not a statistically conclusive experiment.

**Review:** compare all recordings against the original forecast before fitting.
Establish clocks and raw tracking coordinates; inspect gaps, initial state,
contact/intervention, command coverage and end of the supported free-motion
interval. A physical collision is outside the free-cable model. Retain failed
and excluded takes with reasons. A miss is not itself an exclusion reason.

**Round 1:** diagnose cable motion using measured attachment and using the
command-driven drone model on adaptation takes only. Freeze parameter selection
before computing parent/candidate validation diagnostics. Fit only scalar cable damping if that comparison
supports it; preserve drone/residual/EI/Cb and geometry. Keep the best candidate
and stopping reason. Compare M0 and M1 on identical prepared data. Fitting takes
are explicitly in-sample for M1. Validation takes do not enter selection loss.
If initialization or drone response dominates, investigate that issue before
expanding the fit. The proposed small drone-response update needs a new versioned
protocol and comparison support; the current fitter remains cable-only. See
[the integrated adaptation procedure](../development/M0_TO_M1_ADAPTATION.md).

**Prospective check:** freeze a new M1 plan and forecast before recording it.
This is the next evidence for real task performance. If feasible, also repeat
the unchanged reference CSV to distinguish hardware/session drift from a
changed plan. Counterbalance command order within practical flight blocks and
record battery voltage, controller settings, hardware identity and initial
state. Do not infer independent repeats from thousands of tracking frames.

**Round 2 and later:** preserve ancestry and inherited training data. A take used
to fit M1 is also part of M2's training ancestry. Repeatedly inspected validation
takes become development evidence; do not rename them a fresh test. Predeclare
new later flights before updating M2. A persistent reference set can reveal
regression, but cannot remain an untouched final test after repeated tuning.

For a paper, plan a fixed repeat budget after this pilot and report every attempt.
More independent repetitions are needed for a precise success-rate estimate;
three pilot takes cannot support a strong reliability claim. Use per-take paired
differences for identical inputs. With enough independent takes, report an interval
over takes (or session blocks), not over correlated frames. For real hits report
counts, unknown outcomes and a binomial interval only when the trial assumptions
are defensible. No statistical significance or confidence interval is fabricated
by the current UI. Additional seeds/ablations remain separately authorized work.

## Metrics and comparison contract

| Evidence | Main quantities | Interpretation |
| --- | --- | --- |
| Original forecast vs real take | Raw tracked-origin and tip Euclidean RMSE; observed tip-target minimum and coverage | What that frozen forecast predicted before this flight; starting-condition mismatch remains visible. |
| Same-command model comparison | Drone, all observed cable markers and tip RMSE; paired per-take changes | Same causal state, measured inputs, time grid, score interval and masks for every model. |
| Measured-attachment cable diagnostic | Cable/tip RMSE with measured attachment as boundary input | Conditions out drone prediction; cannot alone establish command-driven accuracy. |
| Real task outcome | Reviewed tip-target hit/miss/unknown and feasible/infeasible/unknown execution | A task hit and physical cable collision are distinct. Unobserved tip intervals cannot establish a miss. |
| Data/computation | Whole takes, duration, fitting history, elapsed time and planner settings | Data efficiency and runtime, separate from physical performance. |

RMSE is the square root of the mean squared Euclidean position error in metres;
the UI displays centimetres. All-marker error weights observed marker-time pairs
within a take; the cross-take summary is the arithmetic mean of per-take RMSEs,
giving each take equal weight. It is not pooled-frame RMSE. Coverage is observed
samples divided by samples in the scored interval; initialization frames are
excluded. The saved evaluation also includes cumulative 0.25, 0.5 and 1.0 second
errors only when the reviewed interval supports that horizon. There is no future
measurement correction of the simulated cable during a rollout.

Current task: tip reaches the 5 cm virtual target sphere, with feasible execution.
Sampled minimum distance, prediction RMSE and the planner's reward are different
quantities. A marker sample inside a correctly located sphere supports observed
geometric entry; no sample inside does not rule out between-frame entry. Record
outcome evidence and uncertainty separately. Physical contact timing and its
uncertainty determine the free-motion fitting prefix. Wave/fold style can remain
a diagnostic; it does not silently change the success definition between rounds.

## Using the UI

- **Model table:** M0 and registered completed candidates, their parent, damping,
  model identity and evidence status. Changed/missing artifacts are flagged.
- **Same flight:** choose a saved evaluation, validation/adaptation/all takes,
  and a prediction metric. Lines join paired takes; the dashed line is their
  equal-weight mean. The table shows data use, coverage and interval.
- **Prediction traces:** compare error over time and measured/predicted XZ tip
  paths for the same take. Missing observations remain gaps.
- **Real flights:** original-forecast comparisons, command identities and reviewed
  outcomes. Select a take to open the existing 3D replay. **Review outcome** records
  reviewer/evidence and preserves every earlier catalog/review revision.
- **Fitting progress:** read current reviewed whip jobs, best loss, damping,
  elapsed time and update count. The ceiling bar is not a convergence percentage.

After an authorized fit completes, **Add completed fit** registers it for
comparison only. **Evaluate models on reviewed data** runs all registered models
on one selected prepared job in CUDA, with a progress bar, log and stop between
models. This launches diagnostic rollouts; it does not fit, plan or fly. Loading
the page or opening an existing report never launches work. **Add flight
comparison** imports the report produced by `adapt_whip.py compare`.

The catalog lives at `config/evaluation/campaign.json`, with previous revisions
beside it. New comparison folders under `runs/evaluation` contain curves, masks,
metrics, model/data identities and hashes. Exports include a figure and metadata
identifying the report, metric and selected data role. Audit fixtures are excluded
from the live catalog. Only explicitly registered artifacts appear.

The implemented comparison runner currently supports the reviewed scalar-damping
lineage. It blocks geometry/drone/residual changes until a comparable state/input
mapping is reviewed. This is an explicit scope limit, not permission to silently
reuse the old initialization for an incompatible model.

## Audit changes in this update

The earlier fitter always called a candidate M1; generation indices now propagate
from the frozen parent, enabling M2 and later. New batch setup accepts an explicit
flight selection and checks that package, rather than reusing an M0-specific audit
manifest. Existing prepared protocols, models and original forecasts are unchanged.
Diagnostic coverage now excludes initialization frames outside the scoring window.
Source snapshots still block running old prepared jobs with changed code; prepare
a fresh reviewed job rather than altering its recorded provenance.

Software tests are necessary but do not prove physical accuracy. Current fitting
keeps its conditional diagnostic review, fixed masks, causal history, whole-take
split, bounded CUDA search, independent replay check and separate candidate status.
The planned real M0 flight remains necessary before any justified real M1 update.
