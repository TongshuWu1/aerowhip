# Fresh unseen-system MPPI adaptation check

This is a retained development record, not the current collection checklist.
Read [HANDOFF](../../HANDOFF.md) and [the current experiment protocol](../paper/PAPER_EXPERIMENT_PROTOCOL.md)
for retained-M0 collection and current job status. During the 11 September
cleanup, the retired two-target/swing-and-settle outputs referenced below were
removed. Their old paths and availability statements are historical; they are
not runnable examples or evidence included in the active paper workspace.

## PPO recovery sequence - 11 September 2026

User requested the same post-whip recovery for PPO as MPPI. Shared recovery
already existed; PPO Training progress now defaults Include recovery + CSV on.
Whip-only preview remains an explicit option. No training/reward/physics change.
Frozen best checkpoint at 1,859,584 attempts independently
reproduces score 1914.82511, modeled contact 1.090667 s,
backward drone velocity -1.199 m/s and forward tip velocity 6.421 m/s at contact.
Complete 10.23333 s command/recovery is saved at
`runs/rehearsals_pva/20260911-142454-400090-M2-ppo-1859584-with-recovery`.
Smooth braking/turn, return to [0,0,1.255], final 3 s hover use exactly the same
shared recovery helpers as MPPI. Original learned whip packets are identical;
full predicted drone/cable recovery and envelope checks pass. This is an
intermediate checkpoint while job 20260911-142454-400090 continues; read its
status before reporting. Its final supervisor review remains separate. No
selected physical MPPI/flight/model change. Read audit
`runs/audits/M2-ppo-recovery-20260911`; 15 focused tests pass. Recovery is appended
by the existing trajectory generator, not a phase learned by PPO. No physical
validation is implied. Hover 10 s before CSV and land afterward remain external.

## M2 PPO sustained exploration - 11 September 2026

User authorized the proposed exploration improvement and a larger PPO training
budget. New job `runs/ppo_pva/20260911-142454-400090` is running; read its status
before acting. Do not duplicate/restart. Read `docs/development/M2_PPO_PERSISTENT_EXPLORATION.md`
and `runs/audits/M2-ppo-persistent-20260911`. Fresh actor/critic/optimizer, no copied
MPPI actions or policy weights. Exact full M2, objective, target, limits and
physics retained. Policy now holds jerk for three 30 Hz steps (0.1 s); physics
and contacts still run at 150 Hz and all CSV packets remain 30 Hz. The policy
class/exploration changes; reward is unchanged. One likelihood per decision,
summed intermediate rewards, gamma/lambda=1. Training/inference cadence and
checkpoint timing are verified. 2,048 CUDA environments; minimum 1,048,576
attempts, plateau patience 262,144, review budget 2,097,152. No convergence claim.
34 tests pass/2 missing-fixture skips plus actual-M2 CUDA batch/update, reward,
log-probability, explicit command replay and legacy collection parity. 4,108
protected originals verified preflight. Supervisor automatically reviews the
final best policy and attempts recovery/CSV only after a modeled hit. Preserve
all old policies, selected MPPI, model fits and original real-flight forecasts.
No model fit, promotion or physical flight. Old no-new-training notes below
are historical and superseded only by this single authorized PPO experiment.

## M2 PPO completed - 11 September 2026

Job `runs/ppo_pva/20260911-135019-408009` completed at 188,416 attempts / 92
updates in 647.53 s, stopping by the configured 49,152-attempt plateau rule.
No automatic restart. Best checkpoint is from 139,264 attempts. Independent
best-policy replay and complete recovery passed: modeled hit at 1.08774 s,
forward tip speed 7.5115 m/s, 10.4 s CSV, final rehearsal
`runs/rehearsals_pva/20260911-135019-408009-M2-ppo-whip`.
This is NOT the preferred backward-release whip: drone velocity at contact is
+1.403 m/s versus MPPI -1.517 m/s. PPO/MPPI total scores are 1247.61/1892.43,
fold components 2.72/432.82. Almost all PPO score comes from impact speed.
Same frozen model/reward parity is verified, but MPPI used prior whip command
seeds and PPO started from random weights. Recorded release exploration was
rare (at most 4/2,048 rows in any batch). Exploration/local convergence is the
working diagnosis, not a proven sole cause; no controlled ablation was run.
User requested an explanation, not a new training/reward change. Preserve the
selected MPPI, all policy checkpoints and original forecasts. No fitting,
model promotion or physical flight. Read `docs/development/M2_PPO_MPPI_MATCH.md` and
`runs/audits/M2-ppo-matched-20260911/completion_check.json`; 3,622 protected files
and 284 frozen source files verified. Older running notes below are historical.

## Historical M2 PPO launch and intermediate checks - 11 September 2026

User paused multiple-target work and requested PPO for the selected single-target
M2 MPPI whip, with the same reward/setup. Read `docs/development/M2_PPO_MPPI_MATCH.md`.
Job `runs/ppo_pva/20260911-135019-408009` is running from fresh policy/value/optimizer;
read live status before reporting or acting. No duplicate/restart. Frozen full
M2-frozen-refit-v1 with both residuals, start [0,0,1.255], target [1.25,0,1.00],
1.5 s episode, 2,048 CUDA environments. Exact selected-M2 preferred-fold/impact
reward (1600, scale 4) and bounds; tip contact success; versioned contact-first
best-checkpoint/plateau ranking. No imitation. Matched MPPI/PPO replay score
1892.4306902448288, component/command/termination parity and actual 2,048-row
PPO optimizer smoke pass. 31 focused tests; native progress and live snapshot
replay/refresh pass. Early 24,576-attempt policy preview is a miss, not success.
An 81,920-attempt best policy now independently hits at 1.14964 s, score
1057.73, with checked 10.4 s recovery CSV in
`runs/rehearsals_pva/20260911-135019-408009-M2-ppo-81920-whip`.
It has not learned backward release (drone still forward at contact); fold
score 4.83 vs MPPI 432.82. Training continues unchanged; do not equate contact
or higher tip speed with the desired whip style.
Supervisor `runs/audits/M2-ppo-matched-20260911/run_and_review.py` is already running:
after training it freezes/replays best policy and tries complete recovery/CSV
only for a modeled hit. Consult its status for completion/recovery failures.
Minimum 98,304 attempts, plateau patience 49,152, review limit 500,000; limit is
not convergence. Original MPPI flight/models/data/prior PPO preserved. No fit,
promotion or physical flight. Two-target work is paused; settling stays retired.

## Two-target whip trial — 11 September 2026

User requested one continuous sweep through distinct targets. Read
`docs/development/TWO_TARGET_WHIP.md`. Frozen M2/512 samples/1.5 s, T1 [1.10,-0.15,1.00]
and T2 [1.40,+0.15,1.00] m, start [0,0,1.255]. Jobs
`20260911-133034-712876` and `20260911-133519-124817` completed; both reached
T1 only. Latest closest T2 is 23.04 cm, not a hit. Ordered-contact backend
continues physics after T1; no state reset or collision response. Completion-v2
reward emphasizes both-target progress, with impact credit only after both.
The development live/rehearsal view showed both markers and 1/2 status. Its
review CSV/replay was stored at
`runs/rehearsals_pva/20260911-133519-124817-M2-two-target-whip`; this retired
output was removed during the 11 September cleanup.
54 tests, CUDA/eager/branch parity, independent contact/export/recovery and
native Qt/VTK checks pass. 2,559 protected files unchanged. No fit, promotion,
physical flight or selected-single-target change. Do not duplicate these jobs.
User confirmed the current target positions above are fine. Proposed T2 Y=+0.30
trial was cancelled after bounded seed preflight, before any optimizer job was
prepared or started. Keep current targets; no automatic further run. Settling
remains retired.

## Current scope: whip only â€” 11 September 2026

The user abandoned the swing-and-settle task and requested its removal.
The dedicated planner, configuration, CLI, tests, documentation and MPPI tab
have been removed. Shared whip/PPO replay and MPPI live-view code is restored
to the verified pre-settling implementation. Thirteen settling job/rehearsal/
flight-output directories are archived in place and excluded from active lists;
their original evidence and frozen sources are retained. Do not restore or
restart that task without a new explicit request.

Focus remains the UAV whip system. Selected MPPI is
`20260910-211435-608306-M2-frozen-refit-v1-whip`, using the frozen M2 model.
Whip/PPO commands, forecasts, selections, model fitting and real recordings
are unchanged. No optimization, fitting, model promotion or physical flight
was performed for this cleanup. See `runs/audits/retire-settling-20260911`.
The archived stationary-hover cable-residual diagnostic remains model evidence;
retiring the task does not resolve or erase that limitation.

## Latest paper-pipeline audit â€” 11 September 2026

Read [PAPER_PIPELINE_AUDIT.md](../paper/PAPER_PIPELINE_AUDIT.md) and [PAPER_EXPERIMENT_PROTOCOL.md](../paper/PAPER_EXPERIMENT_PROTOCOL.md) first
for paper decisions. Canonical flown lineage remains M0 â†’ M1-full â†’
M2-frozen-refit-v1, 5/5/3 development takes. Selected paper baseline: staged full
drone/cable/residual simulation-error fitting, offline 512-sample/1.5 s MPPI,
fixed contact-first ranking and soft speed preference. Joint fitting is optional
and unimplemented, not the current method. The release candidate is NOT ready
for clean collection: fresh M0 must use the consistent full model class; actual
sender/metrology and current-state launch readiness need verification. New audit:
88 tests passed/1 skipped, selected-M2 forecast parity and 8/16/32-substep checks,
12 launch-state diagnostics, 646 protected files. No fit, planner, model promotion,
controller or flight-selection change. Preserve all current evidence and raw
roles; no automatic M3. Old text below is historical where it conflicts with
these guides. Audit artifacts: `runs/audits/paper-pipeline-audit-20260911`.


**M2 flight data received and reviewed:** three complete pairs 001â€“003, with
unchanged settings and no contact/intervention confirmed. 001/002 remain assigned
to future adaptation; 003 remains validation. The user completed this batch with
three takes. All 310 selected command rows match per take. Original forecast
drone/tip RMS averages 7.04/9.38 cm; no measured entry into the 5 cm target sphere.
Closest distances are 8.63/6.56/8.84 cm and forward near-approach speeds are
5.76/4.54/5.92 m/s. Read the [M2 flight review](../../runs/data_review/M2-whip-intake-20260910/README.md).
Flights by model â†’ M2-frozen-refit-v1 has all three measured/original-ghost replays.
No new adaptation, planning or selection change followed the review.

**Selected for the next flight:** the user chose rehearsal
`20260910-211435-608306-M2-frozen-refit-v1-whip`. The
[frozen package](../../runs/flight_packages/20260910-211435-608306/README.md) contains
the exact 10.3 s CSV and original M2 forecast. The global flight-selection pointer
now identifies this run; prior pointers/artifacts are preserved. Existing exported
CSV matches exactly. Record paired logs in
`data/flight_batches/M2_whip_20260910-211435-608306/flight_take`.
Predeclared roles: 001/002/004 adaptation, 003/005 validation, names `whip_m2_###`.
Compare this original forecast before reviewing data for any authorized M3 fit.
No physical flight or fitting was performed during selection.

Latest reward follow-up: job `20260910-211435-608306` doubled the contact-speed
weight 800â†’1600. It completed in 67.93 s and retained the exact prior commands,
CSV, forecast and 4.90513 m/s directed contact speed. No further speed improvement.
Separate rehearsal/recovery/native render checks passed; prior artifacts preserved.
See [M2 MPPI results](../development/M2_AGGRESSIVE_MPPI.md). No flight or promotion occurred.

Latest: the user-authorized [M2 aggressive MPPI](../development/M2_AGGRESSIVE_MPPI.md) completed.
Job `20260910-210545-160853` uses the frozen M2 refit with a stronger smooth
contact-speed bonus. Modeled forward hit speed improves 4.51â†’4.91 m/s against
the same-model M1 command baseline. Its separate rehearsal and full recovery
passed and are open for review. No model promotion or real flight occurred.
Earlier no-new-planner notes below are historical.

Latest: [M2 regression analysis](../development/M2_REGRESSION_ANALYSIS.md) is complete. Frozen
component/axis interventions identify a mismatch between separate fitting losses
and command-to-tip prediction, with changed forward motion causing late tip-height
regressions on three takes. No new fitting, planner, promotion or flight followed.

Current: full M1-full â†’ M2 adaptation completed as
`runs/adaptation/M2-full-whip-v1` in 27.8 min; do not duplicate or restart. Read
[the M2 data, fitting and planning contract](../development/M1_TO_M2_ADAPTATION.md).
The user confirmed unchanged hardware/controller/CSV and no contact/intervention.
New 001/002/004 train and 003/005 remain outside fitting/selection. Both inherited
residuals are updated, with prior training replay and post-selection retention
checks. The hard 4 m/s proposal below is superseded by a smooth successful-contact
speed bonus. No new MPPI or flight is launched, and flown M1 evidence is preserved.

M2 remains unpromoted: held-out mean drone/conditional/combined RMS improves
9.18/5.74/6.82â†’6.53/4.58/6.42 cm, but combined tip 005 worsens 4.97â†’6.99 cm.
Across all five new flights combined tip worsens 5.85â†’6.23 cm despite component
improvements. Earlier M0 005 improves; preliminary holdout slightly worsens.
See the linked result for all takes and the saved error-cancellation diagnosis.

Five real M1-full takes have arrived and been compared with the frozen M1 forecast.
Read the [M1 real-flight review](../../runs/data_review/M1-whip-intake-20260910/README.md).
Mean same-command nominal-state drone/tip RMS improves from M0 10.64/18.85 cm
to M1 8.46/8.77 cm; causal-history checks also favor M1. All command rows match.
Near-target outward speed is 5.11 m/s mean, about 16% below the earlier M0 flights;
all current observed paths miss the 5 cm virtual sphere by centre distances
5.87â€“8.54 cm. User requests a 4 m/s minimum at contact for **future M2**; it is
recorded separately, without changing M1. No fit or new planner is running.
cf_3 is identified by unique measured-logger XYZ matching; explicit identity,
hardware/controller/contact confirmations remain pending. M1 real+ghost replay
is open at 001. These facts supersede older no-real-M1 statements below.

Next prospective candidate is prepared in **M1-full**:
[`runs/flight_packages/20260910-181929-716218`](../../runs/flight_packages/20260910-181929-716218/README.md).
Review its saved rehearsal before flight. The 512-sample / 1.5 s search predicts
contact at 1.177714689 s; the complete CSV is 10.333333 s including checked recovery.
The first unchanged-settings M1 run missed; a versioned final-candidate ordering
now prefers feasible tip hits, then the original soft objective. Reward weights,
model, controller and bounds did not change. Read the
[planning comparison and checks](../../runs/audits/M1-full-mppi-contact-20260910/README.md).
19 tests and native CUDA/Qt/VTK checks passed. M0 selection and old artifacts stay
unchanged. This is a separate review candidate, with no physical flight executed.
The next real flight tests the already-fitted M1, then can supply M2 adaptation
data after original-forecast comparison. Older no-new-planner notes are historical.

Current: [full model adaptation](../methods/FULL_MODEL_ADAPTATION.md) completed as
`runs/adaptation/M1-full-whip-v2`; do not duplicate or restart. M1-full is registered
but not promoted. Excluded 005 drone RMS improves 8.93â†’7.64 cm and command-driven
tip 14.89â†’8.81 cm. The result is mixed across takes; no prospective M1 flight exists.
All drone/cable nominal and residual stages are included. The previous partial
M1 stays separate; M0 remains selected. Earlier scope/status notes below are
historical unless repeated by the current full-run report.

The project goal is full drone/cable/residual adaptation; read the current
[model contract](../methods/ADAPTATION_MODEL_CONTRACT.md). The available M1 is only a partial
gain-only candidate, and the raw whip pipeline does not yet train both residuals.
Older scalar-only procedures below are historical trial scopes.

The [first partial real adaptation](../development/M0_M1_FIRST_ADAPTATION.md) is complete. A single drone
response gain was fitted, but held-out errors increased, so the M1 candidate is
not promoted and M0 remains selected. Compare generations contains the saved
M0/M1 same-input results. No fitting/planning is running and no M1 flight occurred.
Older no-M1/synthetic-only statements below are historical.

Five real M0 whip pairs are now received and checked. User-requested split is
001/002/004 adaptation and 003/005 validation. Read the
[intake review](../../runs/data_review/M0-whip-intake-20260910/README.md), including
003's missing initialization marker and estimated clock alignment. Current original-
forecast comparison is `runs/data_review/M0-whip-first-comparison-v2`. No M1 fit or
reinitialized diagnostic rollout has run. Earlier empty-inbox notes are historical.

Latest: [drone-response adaptation review](../methods/DRONE_RESPONSE_ADAPTATION.md) adds
`diagnose-drone` after reviewed preparation. It separates tracking/model error,
estimates native-pose derivatives with gap/edge exclusions, and probes local
gain/delay/attitude-lag sensitivity on adaptation takes only. No acceleration cap
or drone candidate is introduced; the existing prospective fitter remains
cable-only. Windows/RTX 4080 synthetic checks are under
`runs/audits/drone-response-readiness-20260910`.

Latest: [model evolution and evaluation UI](../methods/SIM_REAL_EVALUATION.md) supports
M0/M1/M2 lineage, same-flight prediction comparisons, original-forecast real
results and fitting progress. Open Flight comparison â†’ M0 â†’ M1 â†’ M2.
The live catalog contains only selected M0. Real inbox remains empty; no new
real fitting/planning/flight ran. Reviewed scalar-only scope and prospective
validation remain required. Synthetic audit: `runs/audits/model-evolution-20260910`.

Latest: [M0â†’M1 preparation and audit](../development/M0_TO_M1_ADAPTATION.md) is ready for
reviewed new flight data. Inbox `M0_whip_20260910-022818-648386/flight_take` under
`data/flight_batches` is empty. Original flight package/forecast
unchanged. New raw-coordinate workflow compares the frozen forecast before a
reviewed scalar cable-only update; no automatic neural/drone fitting or model
selection. Timing/contact/history/missingness and whole-take roles are explicit.
No real M1 or physical validation exists yet; synthetic software checks are under
`runs/audits/M0-to-M1-readiness-20260910`. Read that guide before using old fit notes.

Latest: user selected `20260910-022818-648386` for the next measured flight.
Use [selected take instructions](../../runs/flight_packages/20260910-022818-648386/README.md)
and `config/pva/flight_selection.json`; exact CSV and original forecast preserved.
Eleven other MPPI runs and three rehearsals are archived in place and hidden from
active lists; all original files and paths retained. Live-view run binding fixed,
eight UI tests and native VTK saved-snapshot animation check passed. No new
optimization, fit, flight or controller change; prospective real-flight evidence
still pending. See latest HANDOFF and audit `mppi-live-fix-and-flight-selection-20260910`.

Latest: one new MPPI run `20260910-022818-648386` authorized with tip-contact
success. Same previous development M0, 512 samples / complete 1.5 s, reward,
seed, baselines and bounds. Audit `runs/audits/mppi-tip-contact-20260910` owns
the completed supervisor and rehearsal check: 20 updates / 64.48 s, modeled hit
at 1.124878134 s, retained previous sweep's 34-command prefix without improvement.
Full recovery passes through 10.2667 s; new rehearsal saved/selected, 26 frozen
files and 50 prior files verified. No duplicate/restart, fitting or flight.
This supersedes the previous no-new-run note below.

Latest: user authorized simplifying success to a feasible modeled tip hit.
Active MPPI now uses `tip_contact_v1`, existing 5 cm sphere and feasibility
checks. Speed/reversal/wave are diagnostics rather than hard success gates;
the preferred-fold shaping objective/weights remain. Historical missing-version
jobs keep legacy results. Fixed-command CUDA checks pass both saved contacts
and reject miss/invalid controls; no optimizer, fit, flight or replay change.
Read the implementation section in [success review](../methods/WHIP_SUCCESS_CONDITION_REVIEW.md)
and `runs/audits/tip-contact-gate-20260910`. Earlier pause notes are superseded.

Latest: user requested a pause to reconsider papers and the success condition.
Read [success-condition review](../methods/WHIP_SUCCESS_CONDITION_REVIEW.md). Geometric
target contact and our custom wave/speed/reversal requirements must be reported
separately. Both recent motions contact the virtual target; neither establishes
the preferred fold. Review only; no implementation/settings changes, runs or fits.

Latest: the [timing/strength search trial](../development/MPPI_TIMING_SEARCH.md) completed with
no improvement, job `20260910-020448-013542`. Twenty updates / 64.40 s retained
the exact previous-sweep commands; no strong fold or strict success. Checked
rehearsal saved/selected; optimization stopped. Both baselines retained under the same
development M0 and unchanged reward/bounds; 512 samples, full 1.5 s. Read its
status and audit `mppi-timed-baselines-20260910`; no automatic duplicate or fit.

Latest: the [preferred-fold objective and one trial](../development/MPPI_PREFERRED_FOLD.md) completed.
Job `20260910-015017-245988` stopped at plateau after 31 updates / 94.51 s:
1.63 cm closest tip, flatter velocity elevation (17.04 degrees), but no strong
fold or strict strike. Rehearsal/recovery frozen and selected; all optimization
stopped. It uses unchanged development M0 and bounds, a soft
full-sequence shape preference, first-encounter scoring and exact editable old
command timing. Read latest HANDOFF and audit `mppi-preferred-fold-20260910`
status before acting. No new fit, controller change or real flight.

Latest: [preferred-wave review](../methods/WHIP_WAVE_REWARD_REVIEW.md) is complete. The user
wants the old strong travelling fold, not the current target-contact sweep.
Saved-motion comparisons plus two bounded diagnostic replays identify genuine
shape differences and sensitivity to predicted attachment motion. The report
proposes a tested soft fold/propagation objective; no criterion, reward, model,
fit, replay selection or optimizer was changed. Read latest HANDOFF before acting.

Latest outcome: whole-whip job `20260910-011618-458510` completed after 20 updates
at plateau (63.28 s). Closest tip 3.59 cm and first tip contact at 1.12488 s,
but strict success false: dominant-bend stages 0/3. Complete simulated recovery
passed; rehearsal and original forecast saved and selected for playback. Read
latest HANDOFF and [trial details](../development/MPPI_WHOLE_WHIP.md). No optimizer, fit or flight
is currently authorized to start automatically. Launch notes below are historical.

Latest 10 September 01:17: user-authorized [whole-whip MPPI](../development/MPPI_WHOLE_WHIP.md)
trial `20260910-011618-458510` is running; preceding run is stopped at 42 commands.
Same development model; 512 samples, 10 control points, four proposals, adaptive
temperature and separate continuous objective over a complete 1.5 s maneuver.
Strict success and existing bounds unchanged. Check both job status and
`runs/audits/mppi-whole-whip-20260910/status.json`; no automatic alternate run.

10 September 00:43: one explicitly authorized development-M0 MPPI simulation
is running: `runs/mppi_pva/20260910-004200-654955`. Read its live status and
`runs/audits/mppi-development-M0-20260910/status.json` before acting. Original
1,024-sample/2-second settings and editable initialization, new smoothed physical
cable with 0.4/s damping and inherited preliminary drone fit. Live view opened.
No fit-complete/flight-ready claim or global selection change. The one-run worker
will check and freeze a rehearsal after completion if recovery passes; no retry,
flight or M1 fit is automatic. See latest HANDOFF for exact paths and checks.

10 September update: [gradient and damping investigation](../development/PRELIMINARY1_DAMPING_RESOLUTION.md)
completed; a separate unselected M0 development candidate is staged. New real
whip takes will be collected later for the sim-to-real-to-sim loop. Original M0
and saved forecasts remain unchanged; MPPI and neural fitting remain stopped.

Latest status: the user authorized and completed the small
[cable-only pilot](../development/PRELIMINARY1_CABLE_PILOT.md) following the literature review.
Its physical pair is diagnostic only; no new M0 was published or selected.
MPPI `20260909-224132-314704` remains stopped. No drone or residual training ran.
See [method comparison and proposed next steps](../development/PRELIMINARY1_METHOD_COMPARISON.md)
and the latest HANDOFF; older running-status descriptions below are historical.

The new experiment is `20260909-unseen-145g-17g`. The user reports a repaired
145 g drone and 17 g cable assembly. Geometry and tracking-to-attachment convention
are unchanged. Total mass is 162 g. Cable node masses retain the old proportions,
scaled to 17 g; this distribution remains provisional. No old learned model,
recording or policy is active. Historical evidence is recoverable through HANDOFF.

The first five preliminary pairs have now been collected, reviewed and fitted
under the user's subsequent authorization. See [preliminary1 fit details](../development/PRELIMINARY1_FIT.md)
and the job status in HANDOFF. Collection instructions below remain applicable to
additional takes; imports do not automatically enter an existing frozen fit.

## Collect additional takes

Open **Recordings -> Preliminary takes -> Import preliminary take**. Choose one
folder per take containing the original OptiTrack CSV and controller logger CSV.
The files are copied unchanged to `data/raw_takes/<take>/`; imports include the
current hardware identity and source hashes. The original source folder remains.
Keep native recordings and any exact commanded P/V/A trajectory alongside your
collection, plus timing evidence and notes about contact or manual intervention.
Use the same coordinate/attachment conventions; do not assume the tracked origin
is the center of mass. Actual controller gains and recording rigid-body identity
must be documented with the new data, not inferred from old cf7/cf3 logs.

The reset began with an empty inbox and manifest. Importing does not start fitting, planning or
flight. Review new data roles, clock alignment and usable intervals before fitting.
The normalized complete-CSV bootstrap requires reviewed batch inputs; arbitrary
preliminary movements are not automatically a normalized whip batch. Configure the
fit from the actual preliminary recordings after inspecting what was recorded.
The old hard-coded adp0 batch is blocked for this fresh experiment.

## Evidence sequence

1. Collect preliminary motions on the repaired system and review data integrity.
2. Fit fresh M0 from only the selected new preliminary data. Use practical plateau
   stopping and preserve best weights/state; no historical learned initialization.
3. Select M0 for MPPI, optimize, and freeze the full command and predicted motion.
4. Collect a prospective flight of that exact trajectory. Compare against its
   original saved forecast before allowing that flight to update the model.
5. Fit M1 from reviewed adaptation inputs, then generate a new MPPI trajectory.
6. Collect another prospective flight. Keep prior forecasts unchanged and report
   tracking/shape/strike metrics and failures, rather than claiming adaptation from
   training loss alone. Same-command model comparisons separate prediction changes
   from improvements due to a changed MPPI plan.

Current MPPI setup: start [0,0,1.255] m, target [1.25,0,1.0] m,
**Latest user clarification supersedes the trial sequence below:** use original
successful search settings with preliminary M0. Active run
`20260909-224132-314704`: 1,024 samples, 2 s lookahead, initial minimum/patience
20/12, later 5/4. Reward/task/limits and original editable 60-action initialization
match archived parent `20260909-172921-966356`; every prediction and action search
uses new M0. No old accepted trajectory or prediction is passed off as a new result.
The 512-sample sequence below is cancelled and must not restart.

Historical superseded request:
User-authorized trial sequence: 512 samples / 1 s lookahead, then 512 / 1.5 s
if the first completed trial has no valid modeled strike. Both use 30 Hz jerk/PVA
and the separate 5 s maneuver limit. Sequence status:
`runs/audits/mppi-512-1s-then-1p5s/status.json`; check this before starting work.
First job `20260909-223523-507672`, conditional fallback `20260909-223523-778921`.
Manual stop cancels the remaining sequence; no additional trials are automatic.
The user-authorized quick setup uses initial minimum/patience 8/4 and later 3/2,
with no iteration ceiling or planning-time deadline. The stopped 1,024-sample
run retains 14 commands; its continuation `20260909-221253-764801` is now stopped
with 59 commands. The user then authorized the archived farther-target whip as
an editable seed under preliminary M0. Two-second trial `20260909-222142-694207`
was stopped before committing any action. User-requested one-second trial
`20260909-222603-359485` was stopped at 70 commands for the 512-sample sequence.
Each new trial starts at time zero with all 39 original jerk actions:
first 30 initialize the horizon, remaining nine enter as editable tail guesses
as it advances, then zero jerk for the 1 s trial. The 1.5 s fallback sees all
39 seed actions and six zero-jerk tail guesses. Every rollout stays within its
configured horizon; live 3D follows whichever authorized trial is active.
The user prefers an eventual maneuver under two seconds; the current lookahead
change does not enforce that duration. Start/target translate the old geometry by +2 m in X; old model weights,
recovery and prediction ghosts are not imported. Check live status before acting.
Rewards and provisional physical bounds are unchanged. No flight is executed
by this workspace reset. No adaptation improvement has yet been measured.

## Monitor and replay

Use **MPPI -> Optimization progress** for live iterations, recorded elapsed time,
iteration duration, best lookahead distance, sampled hit/infeasible fractions,
effective samples and committed commands. The maneuver bar is simulated time;
the separate activity bar does not imply a planning deadline or an ETA. Switch
between current lookahead search and committed-path charts. The readable console
log is visible by default; disable Follow latest log to inspect older messages.

Use **Rehearsals -> Recorded takes**, or **Recordings -> Replay selected take**,
to play all five preliminary recordings. Play/Pause, a time slider, playback
speed, camera presets and Previous/Next are available; Play all takes advances
through the list. Additional native CSVs can be opened without importing or
modifying them. Raw coordinates/timestamps and missing markers are preserved.
Saved simulated plans remain under Direct PVA rehearsals. Closing and reopening
the UI loads new controls without stopping detached MPPI jobs.

Use **MPPI -> Live 3D search** (or **Watch live 3D**) to watch the evaluated
drone/cable proposals as each batch finishes. Select a candidate, pause/scrub,
change camera/speed, or disable Follow newest iteration to inspect one snapshot.
Faint blue paths show other displayed sample tips; green traces show committed
motion. This is sampled model prediction, not a measured flight. Enable
**Record live 3D candidate snapshots** in setup for new jobs; older frozen jobs
cannot acquire this instrumentation while running. The current continuation has
it enabled. Live telemetry preserves only the latest displayed batch, while
committed commands and optimization history remain saved separately.
