# Fresh M0 iteration workflow

The retained M0 model and flight CSV are unchanged. Use the commands below from
the repository root in PowerShell with `.venv\Scripts\python.exe`. These are the
generic command-line workflow; older M1/M2 one-off runners remain historical.
No new model training or flight-command activation is automatic.

## Ten takes, evaluate before learning

Create a separate collection protocol. This only copies the frozen command and
creates an empty input directory; it does not move recordings from the export.

```powershell
.venv\Scripts\python.exe tools/adapt_whip.py setup --batch data/flight_batches/M0-new --rehearsal runs/rehearsals_pva/20260913-012740-484590-M0-slower-brake-1s --csv exports/M0_Bspline_slower_brake_1s/fullstate_30hz.csv --all-training --take-count 10 --take-prefix M0
```

Keep original raw recordings in the export's `flight_take` folder. Copy the ten
OptiTrack/controller pairs into `data/flight_batches/M0-new/flight_take` as
`M0_001.csv` / `experiment_M0_001.csv` through `M0_010.csv` /
`experiment_M0_010.csv`. Preserve the raw bytes. Complete the existing clock
alignment review and then compare the unchanged forecast:

```powershell
.venv\Scripts\python.exe tools/adapt_whip.py compare --batch data/flight_batches/M0-new --output runs/evaluation/M0-new-before-fit
```

Copy the generated `review.template.json` to a review file, fill in the actual
reviewer, clock, hardware, intervention, contact, and usable free-motion interval
decisions, and keep the predeclared adaptation roles. Do not label unreviewed
trials accepted. Excluded trials stay documented; an exclusion means fewer than
ten usable training takes and must be reported.

For the M4 collection, set `command_source: "event_log"` in the batch protocol
and retain each `experiment_<take>.commands.csv`. Playback and preparation then
use command receipts directly, including commands missed by TF snapshots during
logger gaps. Native OptiTrack still supplies the measured positions. Earlier
protocols keep their existing command source. A final receipt with no observed
hold interval does not extend command coverage beyond the log.

```powershell
.venv\Scripts\python.exe tools/adapt_whip.py prepare --batch data/flight_batches/M0-new --comparison runs/evaluation/M0-new-before-fit --review runs/evaluation/M0-new-before-fit/review.json --job runs/adaptation/M0-new-inputs --full-model
.venv\Scripts\python.exe tools/adapt_whip.py prepare-full --whip-source runs/adaptation/M0-new-inputs --preliminary-source runs/adaptation/20260909-preliminary1-M0-v2 --job runs/adaptation/M1-new --parent-id M0 --candidate-id M1 --all-training
.venv\Scripts\python.exe tools/adapt_whip.py fit-full --job runs/adaptation/M1-new
```

Before any fitting update, `fit-full` verifies the parent catalog identity, checks
registered raw-source ancestry, and saves `before_update/report.json` with frozen
parent predictions. The ancestry check is limited to recorded hashes; it does
not establish independence from earlier method development. The comparison of
the originally saved forecast remains separate from reinitialized predictions.

All accepted current takes then enter training. The parent parameters and neural
weights are warm-started; each stage creates a fresh optimizer. Post-fit scores
on those takes are training diagnostics. With no validation takes, registration
succeeds as a candidate awaiting next-batch evaluation, without claiming a
validation improvement or automatically promoting the candidate.

For later rounds, use the next flown rehearsal and new batch/output names, set
the corresponding `--parent-id` and `--candidate-id`, and repeat
`--prior-whip-source` for earlier prepared whipping jobs that should enter replay.
Earlier training takes are reported as retention diagnostics. Legacy holdouts,
including the preliminary holdout, retain their roles and remain outside replay.

The residual stages use the existing plateau rule by default. A JSON supplied
through `--contract` can freeze `stage_budgets` for **both** `drone_residual` and
`cable_residual`, each with positive `maximum_updates` and `maximum_seconds`.
Keep that contract consistent across model generations. A budget stop is logged
as a budget stop, not convergence; final numerical checks are outside its time
limit. Prepared jobs bind code hashes: finish code changes before preparing a
real training job. Never rewrite old hashes to bypass that check.

## Cable loss for the next update

Starting with the next update after M4, new default contracts set
`cable_tip_weight: 0.0`: all valid tracked-marker samples have equal weight,
including the tip once. This applies to both physical-parameter and
neural-residual fitting. Missing samples remain masked; regularization is unchanged.
The weight is saved in each prepared job's contract. M1–M3 used a 50% all-marker
and 50% extra tip loss; M4 used 75% all-marker and 25% extra tip loss.
Historical contracts without this field retain the 50/50 rule; when reusing an
old contract for the next update, explicitly set `cable_tip_weight` to `0.0`.
Training loss values across this change are not directly comparable.

## Fitting checks

The full residual finite-difference check covers every trainable tensor and two
deterministic joint directions. Two adjacent perturbation sizes must agree with
autograd and each other using absolute and relative tolerances. Near-zero
derivatives can pass if the numerical derivatives agree. Parameter values are
restored even when a probe raises an exception.

The cable residual also compares two recurrent CUDA graph blocks against the
same eager equations, for outputs and every parameter gradient. This separates
capture errors from the full-window numerical sensitivity test; it is not proof
of physical model accuracy.

## Trajectory correction

```powershell
.venv\Scripts\python.exe tools/correct_reference.py --model-job runs/adaptation/M1-new --reference runs/reference_tracking/M0-paper-fixed-reference --job runs/reference_tracking/M1-new-local --output runs/rehearsals_pva/M1-new-local --export exports/M1_new_local
```

All three output directories must be new. The fitted model must be complete and
pass its saved hashes. For a later round, `--previous-rehearsal` warm-starts the
last corrected controls while keeping the original physical reference fixed.
The active-flight pointer is not changed by this command.

The average fixed-time tracking objective and physical limits remain unchanged.
The optimizer adapts damping and trust radius using actual versus predicted
cost reduction. A single small backtracked step no longer declares convergence.
Stopping for small progress requires repeated qualifying improvements and a
command-constrained stationarity check. Exhausted iterations/radius are reported
separately. Full recovery, command limits, and independent production replay
checks still precede export.

Every evaluated line-search candidate records planned-time reference error and
target distance in `history.json`, including rejected candidates. Candidates
rejected before a rollout have null strike metrics and an explicit reason.
The optional `--strike-guard target` or `--strike-guard reference` prevents the
chosen planned-time metric from exceeding its value at the starting command;
`--strike-tolerance-m` defaults to zero. The guard defaults to `none`: enabling
it changes the feasible set and must stay consistent in the experiment protocol.

## Initial-state sensitivity

After raw takes have been reviewed and prepared:

```powershell
.venv\Scripts\python.exe tools/check_initial_state_sensitivity.py --model runs/rehearsals_pva/20260913-012740-484590-M0-slower-brake-1s/model.json --reference runs/reference_tracking/M0-paper-fixed-reference --prepared-job runs/adaptation/M0-new-inputs --output runs/evaluation/M0-new-initial-state
```

This compares nominal, measured-shape, measured-relative-velocity, and combined
cable initializations under the same model and command. Causal pre-command
states are translated to the planned root, root velocity is subtracted, and each
scenario is projected onto cable length/velocity constraints, so the vehicle
stays settled. Those transformations and projection magnitudes are recorded:
these are counterfactual sensitivity scenarios, not exact flight replays.
Invalid rollouts are unscored. No fitting data or model physics are changed, and
an ensemble correction objective has not been introduced.

Physical performance and statistical significance must still come from the new
collections, with the take as the sampling unit and a consistent evaluation
protocol. Passing these software checks cannot establish either conclusion.
