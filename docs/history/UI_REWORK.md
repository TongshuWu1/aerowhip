# Research UI rework — 2026-09-05

The desktop shell now contains Baseline & Data, Task & Rewards, PPO, SAC, and
Real Data & Adaptation. MPCC, the global replay page and the comparison page
are removed from navigation. Their legacy source utilities remain available.

## Baseline and data

The first page manages preliminary calibration recordings. Imported folders
must contain exactly one recognized logger CSV and one Motive CSV. Existing
raw takes are never overwritten. Enable/disable and fit/validation roles are
saved in the dataset manifest. The protected test role is locked in this UI.

Processing and fitting run as background processes. A job snapshots the model,
data-role manifest and fitting configuration in `data/workflow_jobs/<id>/`.
Fitting reconstructs its own force/state dataset, compares against the supplied
current EI/Cb, selects using fitting takes only, and evaluates validation takes.
It records marker/tip errors and measured/active/candidate trajectories for the
first validation window. The attachment boundary uses measured motion.

A completed fit remains a candidate until applied. Applying writes a baseline
version under `data/baselines/` and updates `config/model.json`. Manual parameter
application is labeled as manually configured provenance. The prior baseline
calibration artifact is preserved. Physics inputs and job controls are locked
while that page's processing/fitting job runs.

## Task and training

Task & Rewards retains simple hit gates and reward weights, adds target/direction,
initial attachment, duration and force limits, and keeps shaping scales collapsed.
Apply writes the shared task/PPO reward settings and saves a version. Existing
run snapshots do not change. The time penalty is still 1 point per strike second.

Algorithm pages expose run selection, seed, target attempts, batch, device and
the main algorithm-specific tuning controls. New runs use the currently applied
model/task. Continuation inherits parent snapshots and creates a new directory;
SAC restarts replay collection while restoring its networks and optimizers.
The UI's validation cadence is one evaluation after every collection batch.
This adds validation compute, particularly for SAC compared with its old CLI
default cadence. Collection and validation batch sizes are independent.

Learning plots show raw current-policy full-set validation, while the 3D viewport
plays its first actual trial. It performs no extra background evaluation. PPO
rollback keeps the accepted trial data and labels reuse. A journal entry includes
the training-attempt count, original evaluation ID, checkpoint bytes/hash,
configuration hashes, scenario arrays/hash, all trial outcomes and a first-trial
recording. Duplicate writes at the same attempt/checkpoint are ignored.

Run latest policy snapshots `latest.pt`, never silently substitutes the best
policy, and uses that run's saved model/task. It settles in PID hover, plans from
the initial drone/cable estimate, executes the force sequence once, and returns
to PID at its scheduled cutoff. Actual contact only affects displayed/scored
outcomes. The live simulation continues hovering until stopped.

Closing the UI does not stop training/fitting processes. Live simulation workers
are stopped and allowed to finish before their Qt resources are destroyed.

## Figures and remaining research work

Learning export uses fixed figure dimensions independent of the window, with
PDF/SVG, 600 dpi PNG, source CSV and metadata. It reports one training seed and
does not invent error bars. Fitting exports include measured/predicted window
CSV and the full numerical fit report. The plots remain empty until actual
evaluation results exist.

Real Data & Adaptation is intentionally a reserved page for further discussion.
Multi-seed orchestration/aggregation, common final-test checkpoint selection,
hardware initial-state estimation and measured-shape training coverage are not
claimed to be implemented by this UI rework.

Validation includes small real PPO/SAC training runs in temporary test directories,
an actual preliminary-data fitting smoke test, saved-evaluation identity/reuse,
baseline versioning, protected test-role controls, and rendering at 1440×900 and
1120×720. No paper training run or new production fit was launched.
