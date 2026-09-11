# Project instructions

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
before acting. Do not duplicate/restart. Read `docs/M2_PPO_PERSISTENT_EXPLORATION.md`
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
model promotion or physical flight. Read `docs/M2_PPO_MPPI_MATCH.md` and
`runs/audits/M2-ppo-matched-20260911/completion_check.json`; 3,622 protected files
and 284 frozen source files verified. Older running notes below are historical.

## Historical M2 PPO launch and intermediate checks - 11 September 2026

User paused multiple-target work and requested PPO for the selected single-target
M2 MPPI whip, with the same reward/setup. Read `docs/M2_PPO_MPPI_MATCH.md`.
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
`docs/TWO_TARGET_WHIP.md`. Frozen M2/512 samples/1.5 s, T1 [1.10,-0.15,1.00]
and T2 [1.40,+0.15,1.00] m, start [0,0,1.255]. Jobs
`20260911-133034-712876` and `20260911-133519-124817` completed; both reached
T1 only. Latest closest T2 is 23.04 cm, not a hit. Ordered-contact backend
continues physics after T1; no state reset or collision response. Completion-v2
reward emphasizes both-target progress, with impact credit only after both.
Live/rehearsal show both markers and honest 1/2 status. Latest complete review
CSV/replay is `runs/rehearsals_pva/20260911-133519-124817-M2-two-target-whip`.
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

Read `docs/PAPER_PIPELINE_AUDIT.md` and `docs/PAPER_EXPERIMENT_PROTOCOL.md` first
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


Read [HANDOFF.md](HANDOFF.md) first. For the active experiment, read
[docs/NEW_SYSTEM_CHECK.md](docs/NEW_SYSTEM_CHECK.md). For paper writing, read
[docs/PAPER_WRITING_HANDOFF.md](docs/PAPER_WRITING_HANDOFF.md) and
[docs/PAPER_READINESS_REVIEW.md](docs/PAPER_READINESS_REVIEW.md).

## Current scope

- Latest system comparison/UI audit: read docs/M0_M1_M2_SYSTEM_COMPARISON.md and
  runs/evaluation/M0-M1-M2-system-review-20260910-v2. Actual flown lineage is M0 â†’
  M1-full â†’ M2-frozen-refit-v1; all three have real data (5/5/3 takes). Older bullets
  below saying M1/M2 do not exist/have not flown are historical. New matched M2
  flight diagnostics give complete tip RMS 14.86/7.99/7.03 cm; none of these models
  trained on M2 data, but M2 worsens take002. Own original forecasts give tip means
  17.17/8.77/9.38 cm; observed virtual entries 0/5,0/5,0/3. Development evidence only.
  Study overview UI separates these questions and verifies evidence hashes; no
  automatic compute. 36 complete frozen predictions plus conditional diagnostics,
  exact old-report parity, 231 original files unchanged, 19 tests/native Qt/VTK
  checks. No fitting, planner, controller, model promotion or flight selection change.
  Preserve all variants and original ghosts; initial-state robustness remains
  discussed future work. Report's exclusions follow model training ancestry.

- Three prospective M2 takes are reviewed and registered under M2-frozen-refit-v1.
  Read runs/data_review/M2-whip-intake-20260910/README.md. User confirms complete
  batch 001â€“003, unchanged selected CSV/settings/hardware/origin, no contact or
  intervention. 001/002 adaptation, 003 validation; planned 004/005 absent.
  310/310 logged command rows match exactly per take. Original-forecast mean
  drone/tip RMS 7.04/9.38 cm; 0/3 entries into 5 cm virtual sphere, closest centre
  distances 8.63/6.56/8.84 cm. Forward near-approach speed 5.76/4.54/5.92 m/s.
  Raw data, forecasts/models and global flight selection preserved. Nine tests
  and native Windows Qt/VTK paired playback pass. No fit/planner/model promotion
  or physical flight command. New M3 fit requires explicit authorization and
  strict data preparation under the frozen staged method; do not auto-start.

- Latest user selected exact screenshot rehearsal
  20260910-211435-608306-M2-frozen-refit-v1-whip for use. Global flight_selection
  now binds that M2 run; old pointer archived. Package runs/flight_packages/
  20260910-211435-608306; complete CSV 10.3 s, original forecast frozen.
  Existing M2_whip_20260910-211435-608306 user-exported CSV matched and is preserved.
  Its empty flight_take inbox and raw protocol are ready: whip_m2_001/002/004
  adaptation, 003/005 validation, predeclared before flight data. Full staged
  method planned only after reviewed data/explicit fit authorization, not legacy
  scalar-only defaults. Audit M2-flight-selection-20260910 verifies 2,399 protected
  files; authorized selection pointer is the sole changed protected file. No fit,
  planner, physical flight or model-validation promotion. Preserve old artifacts.

- Latest requested increase: impact weight 800â†’1600. Job
  runs/mppi_pva/20260910-211435-608306; audit M2-impact1600-20260910.
  Completed at 20 updates / 67.93 s, plateau. Same 35 commands, complete CSV,
  forecast and 4.90513 m/s contact speed; no further speed gain. Independent
  replay/recovery/native frames pass; 2,372 protected originals unchanged.
  Separate rehearsal open paused at start. Do not restart.
  Read status; no duplicate. Same M2/512/1.5 s and all other numerical settings.
  Prior M2 winner is editable incumbent; M1/M0 baselines/reference bytes retained.
  CUDA preflight reproduces prior 4.90513 m/s; four impact tests pass. Separate
  live/rehearsal/recovery; preserve previous evidence and selections. No fitting,
  model promotion or physical flight. See docs/M2_AGGRESSIVE_MPPI.md.

- Latest user requests a harder-hit MPPI with frozen M2. Job
  runs/mppi_pva/20260910-210545-160853; audit M2-aggressive-mppi-20260910.
  Completed in 85.11 s / 26 updates; modeled hit at 1.152791 s. Same-model directed
  hit speed 4.50946â†’4.90513 m/s (+8.77%); 35-command whip, 10.3 s full CSV.
  Independent replay/contact-speed/recovery and native live/rehearsal checks pass;
  1,957 protected files unchanged. Rehearsal open paused at start. Do not restart.
  Read status; no duplicate. See docs/M2_AGGRESSIVE_MPPI.md. 512 samples / 1.5 s,
  successful forward tip-speed bonus weight 800, scale 4 m/s; no speed cutoff or
  earlier-hit reward. Editable M1/M0 command seeds are rerolled with M2. Same-model
  M1 seed predicts contact at 4.50946 m/s. Live 3D and separate rehearsal/recovery.
  No refitting, promotion or physical flight; preserve frozen fits/CSV/ghosts and
  global configuration. Sixteen focused tests and CUDA baseline checks passed.

- Latest user authorized a documented, frozen staged-method M2 refit. Read
  docs/FROZEN_SYSTEM_IDENTIFICATION.md and runs/audits/M2-frozen-refit-v1.
  Job runs/adaptation/M2-frozen-refit-v1 starts from exact M1-full, same reviewed
  data and numerical settings as old M2; only the candidate ID changes.
  Completed in 1,656.6 s on Windows/RTX 4080. Parameters, NN tensors, training-loss
  histories, model signature and per-take RMS exactly reproduce old M2. Gradient
  probes/runtime parity/native UI passed; 739 protected files unchanged. Combined
  tip prediction remains mixed; candidate not promoted. See audit README/summary.
  No duplicate/restart. Combined fitting remains proposed, not part of this refit.
  Preserve old M2 and all flight/model evidence. No promotion or
  MPPI/flight change. Current runs remain development evidence.

- User explicitly clarifies the paper contribution is the whole UAV open-loop
  dynamic-whipping system using real-to-sim-to-real updates and MPPI, not a novel
  adaptation algorithm. Immediate need: one repeatable identification procedure.
  docs/SYSTEMATIC_ADAPTATION_PROTOCOL.md v0.2 specifies parent-initialized weighted
  regularized refitting, staged component fitting plus proposed combined refinement.
  Adaptation-algorithm/cold-refit benchmarks are optional for specific system claims,
  not mandatory novelty work. Current data/runs remain development; separate clean
  study follows method freeze. No new implementation or fit/planner/promotion.

- Latest user wants one systematic paper adaptation method and clarifies that
  current runs/data are development; a separate clean experiment comes after
  refinement. Read docs/SYSTEMATIC_ADAPTATION_PROTOCOL.md (design v0.1 only).
  Proposed common objective includes drone, measured-boundary cable and complete
  command-to-cable loss with regularization; staged initialization then combined
  refinement. Freeze method/planner/evaluation before new collection; fresh M0
  uses new preliminary recordings and reset learned weights/optimizer. Current
  artifacts and fitting roles remain unchanged. No new fitter, fit, planner,
  promotion or deletion; proposal is not a running-job authorization.

- Latest user requested careful explanation of M2's mixed combined prediction.
  Read docs/M2_REGRESSION_ANALYSIS.md and runs/audits/M2-regression-analysis-20260910.
  Frozen component/XYZ interventions identify changed forward attachment motion
  as the main late tip-height regression mechanism on 001/004/005, helping 003.
  Separate fitting losses omit command-to-tip optimization; old error cancellation
  also matters. 97 protected files unchanged, own-model parity within 1.83e-12 m.
  Coupled final refinement with component retention is proposed only. No new fit,
  MPPI, model promotion, controller/command change or flight; preserve M1/M2.

- Latest user authorizes full M1-full â†’ M2 adaptation and replaces the proposed
  hard 4 m/s minimum with stronger soft impact-speed reward. Read
  docs/M1_TO_M2_ADAPTATION.md and runs/audits/M2-full-adaptation-20260910.
  Job runs/adaptation/M2-full-whip-v1 completed in 27.8 min on Windows/RTX 4080;
  do not duplicate or restart. M2 registered, not promoted. 42 tests and native
  comparison UI checks pass; 70 protected files unchanged. Both selected NNs
  pass temporal gradient checks; drone 260/cable 200 updates stopped by plateau.
  Held-out 003/005 mean drone/conditional/combined tip RMS is
  9.18/5.74/6.82â†’6.53/4.58/6.42 cm, but 005 combined tip worsens 4.97â†’6.99 cm.
  All five new takes: components improve, combined tip worsens 5.85â†’6.23 cm.
  Prior M0 005 improves; preliminary holdout slightly worsens. Do not claim
  uniform combined improvement. UI opens M2-full-whip-v1 comparison. No refit
  selected from validation; exact error-cancellation decomposition is saved.
  User confirmed unchanged hardware/controller/CSV and no contact/intervention.
  New M1 001/002/004 train, 003/005 are development holdouts. Preserve earlier
  splits. Both NNs warm-start from M1-full; prior M0 training and preliminary
  training replay remain included. Candidate generation is 2, parent M1-full.
  Neural stopping has no routine ceiling; numerical verification and held-out
  retention followed frozen selection. No automatic promotion or planner/flight launch.
  Future config/pva/m2_mppi.json uses 400*vÂ²/(16+vÂ²) at feasible tip contact,
  without a speed cutoff. Flown artifacts and global MPPI/PPO/flight selections
  stay unchanged. Older unconfirmed/hard-minimum/no-fit bullets are historical.

- Five real M1-full pairs are processed. Read
  runs/data_review/M1-whip-intake-20260910/README.md and
  runs/data_review/M1-whip-first-comparison. All 311 logged CSV rows match exactly;
  cf_3 uniquely matches measured logger XYZ, cf_6 does not. Explicit user identity
  and controller/hardware/contact/intervention confirmations remain pending.
  Same-command nominal initial-state M0â†’M1 drone RMS 10.64â†’8.46 cm, tip
  18.85â†’8.77 cm; separate causal-history drone 9.35â†’8.74, conditional tip
  9.20â†’5.63, coupled tip 15.98â†’5.85 cm. Raw XYZ/gaps preserved; clock estimated.
  Near-target measured outward speed 5.11 m/s mean versus previous M0 6.06 m/s,
  on different commands. No observed entry into unchanged 5 cm virtual sphere;
  nearest centres 5.87â€“8.54 cm including first 0.5 s recovery. No physical-contact
  assertion. User wants hard minimum 4 m/s outward tip speed at contact for M2
  only: config/pva/m2_planning_requirements.json records it; no M2 planner or
  historical M1 criterion change. No fit/new planner/model promotion; M2 roles
  unassigned. All five are in Flight comparison â†’ M1-full with original ghost;
  app/replay startup at 001. Catalog binds exact signature/ID, not gain-only M1.
  37 tests, native Windows/RTX 4080 checks passed; 11 uploaded CSVs / 80 prior
  protected files unchanged, catalog revisions preserved. Prior no-real-M1 notes
  are historical. Do not fit or promote from this intake automatically.

- Latest user requested MPPI for the next real flight. Completed M1-full candidate
  job runs/mppi_pva/20260910-181929-716218; no duplicate/restart. Same frozen full
  model, both NNs, old objective/seeds/bounds, 512 samples / 1.5 s. First M1 run
  20260910-181640-773710 preferred a 7.09 cm miss despite sampled hits and is
  preserved. Follow-up uses versioned tip_contact_then_score_v1: feasible hits
  first, original score among hits. No reward-weight/controller/model/PPO change.
  31 updates / 102.35 s plateau; modeled contact 1.177714689 s, whip 1.2 s, complete
  CSV 10.333333 s including checked recovery. Read audit
  runs/audits/M1-full-mppi-contact-20260910/README.md. Review package/CSV/forecast:
  runs/flight_packages/20260910-181929-716218. App open at matching M1-full rehearsal.
  19 tests, independent native CUDA replay/export parity and Qt/VTK scene checks
  passed; 53 protected previous files unchanged. M0 remains global selection;
  new candidate is separate, not promoted. No physical M1 flight or new fitting.
  Next real flight evaluates the already-fitted M1-full, then supplies possible
  M2 data after original-forecast comparison. Disclose planner selection change
  alongside model adaptation; do not claim a pure model-only task improvement.

- User authorized implementing AND running full drone/cable/residual adaptation.
  Completed continuation runs/adaptation/M1-full-whip-v2; no duplicate/restart.
  M1-full registered, not promoted. 005 drone 8.93â†’7.64 cm, conditional tip
  6.74â†’3.95 cm, coupled tip 14.89â†’8.81 cm; 004 coupled tip worsens 10.40â†’10.98 cm.
  Preliminary holdout drone mean RMS 9.81â†’7.80 cm, conditional tip 4.94â†’4.92 cm.
  Both NNs stopped by plateau (drone 585, cable 280) and passed numerical checks.
  V2 continuation/validation 1950.1 s; prior V1/audit work separate. 38 tests and
  frozen source/data/M0/flight checks passed on Windows/RTX 4080. M0 stays selected.
  V1 failed at cable NN gradient verification before any cable NN update; preserve
  it. V2 resumes update-400 Adam state with no routine neural ceiling, and retains
  the best numerically verified three-parameter cable iterate. All windows and
  physics unchanged. Read docs/TRAINING_HORIZON_RESIDUAL_REVIEW.md; user asks for
  actual stage/removal comparisons, longer-window reasoning and primary papers.
  Latest user clarifies all hard limits should be judged by effectiveness, not
  removed blindly. Nominal bounds/80-evaluation guard did not bind. Residual
  Â±0.5 m/sÂ² is a tunable discrepancy bound, never an identified vehicle limit.
  Removal diagnostics: drone NN off improves 005 drone/tip to 6.26/8.21 cm but
  worsens preliminary drone to 12.71 cm (7.80 with NN). Cable NN improves 005
  conditional tip 6.58â†’3.95 and coupled tip 11.57â†’8.81; preliminary conditional
  tip worsens slightly versus adapted physics alone, 4.82â†’4.92. Full candidate
  retained, no removal variant selected; universal necessity is not established.
  Both NN bounds are occasionally active (~6% of sampled training components).
  UI lineage now shows actual sibling-parent edges, not a false M0â†’M1â†’M1-full chain.
  Read docs/FULL_MODEL_ADAPTATION.md. Full GPU gradient/production checks passed;
  all six gains, delay, attitude, drone NN, EI/Cb/external damping and cable NN
  are included. 001/002/004 plus preliminary training replay; 005 and preliminary
  holdout after frozen selection. Prior M1 remains separate; new ID M1-full is
  another child of M0, not M2. No automatic promotion, MPPI or flight.
  Earlier review-only/missing-implementation/scalar-only notes are historical.

- User corrected the adaptation scope: improve the reusable drone AND cable model,
  including residual corrections. Read docs/ADAPTATION_MODEL_CONTRACT.md first.
  Current M1 is a partial gain-only candidate, not completed full adaptation.
  Preliminary code has nominal/residual training for both subsystems; raw whip
  CLI exposes only separate scalar fits and needs a combined staged workflow.
  Audit runs/audits/M1-regression-audit-20260910 verifies raw reconstruction,
  reference integration, per-axis/attitude regressions and clock scenarios.
  Latest work is review/documentation only; no new fit, selection or planner.
  Keep M0 and all evidence. 005 remains validation for the first frozen candidate,
  but its reuse in method development must be disclosed. Do not change data roles
  or force every parameter to change merely to claim full adaptation.

- First real adaptation completed. Read docs/M0_M1_FIRST_ADAPTATION.md and
  runs/audits/M0-real-adaptation-20260910. Job runs/adaptation/M1-whip-response-v1
  fits only drone feedforward_xy 0.74668â†’0.61834 under reviewed response contract.
  Cable/NN/delay/controller/commands fixed. Five updates / 57.3 s, plateau. Candidate
  M1 registered but NOT promoted: held-out 005 drone RMS 8.93â†’10.41 cm and tip
  14.89â†’17.93 cm worsen. M0 stays selected; no automatic retune/new fit/planner.
  001/002/004 adaptation; 005 held out; 003 reserved but strict initializer unusable.
  Compare generations loads runs/evaluation/M0-M1-whip-response-v1. 34 tests and
  Windows/RTX 4080 parity/source checks passed. No physical M1 flight. Earlier
  cable-only/not-implemented/only-M0 catalog statements are historical.

- Latest UI request organizes by model generation. Flight comparison â†’ Flights by
  model and Recordings share generation/session filtering; replay takes show roles.
  Compare generations is separate. Only M0 exists; M1 is an empty future state.
  Existing M0 comparison v2 is registered in campaign.json (prior revision retained).
  No model/data/forecast/flight selection changes, fits or rollouts. 27 tests and
  native Windows/RTX 4080 generation switching/playback passed. Read
  runs/audits/flight-generations-ui-20260910 and docs/SIM_REAL_EVALUATION.md.

- Latest user requested measured M0 flight with original MPPI ghost. Existing
  Flight comparison renderer is connected; all five takes are available in
  Real flight + plan ghost. config/pva/flight_replay.json opens 001 at quarter speed
  on startup. Automatic ghost selection binds batch protocol and forecast hash;
  shortcut loads immediately. Fourteen tests and native paired VTK playback passed.
  Audit: runs/audits/M0-flight-ghost-ui-20260910. Raw/M0/CSV/flight selection unchanged;
  no fitting or planning. Keep the original forecast, missing-marker gaps and
  estimated-clock qualification. App was left open paused at start.

- Five measured M0 whip pairs have arrived. Read
  runs/data_review/M0-whip-intake-20260910/README.md and current comparison
  runs/data_review/M0-whip-first-comparison-v2. User requests 001/002/004 adaptation,
  003/005 validation; same selected CSV/settings, no contact confirmed. Logged
  packets match all 309 CSV rows in every take. Clock alignment is estimated.
  003 has missing c5 initialization/whip samples and is reserved validation evidence,
  not eligible for the current strict cable initializer. 005 history is usable.
  Raw/M0 unchanged; no M1 fit or diagnostic rollout. Empty-inbox notes are historical.

- Latest user retained original selected MPPI `20260910-022818-648386` with its
  rehearsal, plus PPO `20260910-115619-908807` and latest saved preview
  `20260910-173200-563888-ppo-preview` (latest.pt, 491,520 attempts, preview-only).
  Seven other entries, including the harder-hit rerun, were archived in place;
  5,970 existing files verified unchanged. Audit:
  runs/audits/keep-latest-ppo-selected-mppi-20260910/result.json.
  Keep these active; do not restore archived entries automatically. Original MPPI
  flight selection/CSV/forecast unchanged. No rollout or deletion occurred.
  Future harder-hit reward settings were not reverted by this library cleanup.

- User corrected the request to harder impact, not earlier time. Earlier-hit
  prepared job 20260910-135950-415845 was cancelled without starting. Future MPPI
  settings use impact=200 and impact_scale_m_s=4, no early-hit fields. A bonus
  200*vÂ²/(16+vÂ²) uses outward tip speed at actual successful contact; no new cap.
  Corrected authorized run: 20260910-140256-116887, same M0/512/1.5 s/search.
  Completed at 20 updates / 64.87 s; exact old 34-command motion retained, directed
  tip impact speed 4.17886 m/s, no improvement. Separate rehearsal/recovery passed;
  complete CSV bytes match the original. Score increase is new bonus only.
  Read runs/audits/mppi-harder-hit-20260910/status.json; do not duplicate/restart.
  Preserve original flight selection/CSV/forecast and PPO. Separate rehearsal
  generation is authorized for this run, not automatic flight selection.

- Latest user authorized an earlier-hit MPPI reward. Future config/pva/mppi.json
  adds reward.early_hit=100 and early_hit_scale_s=1; successful feasible entry gets
  100*exp(-absolute hit time/1s) once, no miss/failure credit. No horizon-dependent
  normalization or hard time limit. Historical missing keys mean zero. PPO and
  selected M0 flight CSV/forecast/settings remain unchanged. 23 tests and a bounded
  Windows/RTX 4080 fixed-command check passed. No new optimizer/flight was started.
  Read runs/audits/mppi-earlier-hit-20260910 before interpreting old scores.

- Latest user authorized adaptation preparation and literature/math review while
  collecting M0 whip data. Read docs/DRONE_RESPONSE_ADAPTATION.md and audit
  runs/audits/drone-response-readiness-20260910. `tools/adapt_whip.py diagnose-drone`
  is implemented: adaptation-only recursive pose/derivative/sensitivity diagnostics,
  no candidate selection or saturation fit. 37 tests and synthetic CUDA/math checks
  passed. The scalar cable fitter remains the only implemented prospective fit;
  changed-drone publication/comparison needs a reviewed extension after diagnostics.
  No hard command-acceleration cap, M0/CSV/forecast/controller change or real fit.

- Latest user clarification: learn drone command-to-executed-motion capability
  during adaptation; do not add a hard generated-acceleration cap from the
  preliminary command maximum. Preserve selected M0/CSV/forecast and existing
  checks. Read docs/M0_TO_M1_ADAPTATION.md and integrated-adaptation-20260910
  audit. Pre-fit model diagnosis now uses adaptation takes only; validation
  diagnostics follow frozen parameter selection. Drone response/saturation
  updating is proposed, not implemented by the current scalar cable fitter.
  Actual sender/logger semantics remain unverified. No real fit or flight ran.

- Latest user requested a second audit of drone learning and commandâ†’predicted
  trajectory execution. Read docs/DRONE_COMMAND_CHAIN_AUDIT.md and
  runs/audits/drone-command-chain-20260910. Internal chain and original bytes
  verified. Selected acceleration extrapolates (14.31 vs 8.18 m/sÂ² training).
  Actual forthcoming flight sender/logger source still unidentified; do not
  claim confirmed onboard receipt or physical accuracy. No model/fit/flight/
  controller change. PPO has meanwhile completed at 491,520 attempts by plateau;
  best fixed-scenario score 692.8127 with modeled contact. Do not restart.

- Latest user authorized PPO with the same development M0 and actual objective as
  selected MPPI, while retaining MPPI for prospective flight data. Read
  docs/PPO_MPPI_OBJECTIVE.md and runs/audits/ppo-mppi-objective-20260910.
  Trial runs/ppo_pva/20260910-115619-908807 starts from scratch, 2,048 CUDA
  environments, 1.5 s, exact start/target, shared preferred-fold score and
  tip_contact_v1. It does not copy MPPI commands. Read live status first; no
  duplicate/restart. PPO authorization supersedes older PPO stop notes only.
  MPPI flight CSV/forecast/model/selection remain frozen; no fitting or flight.

- Latest user requested theory/implementation audit and unused-source cleanup.
  Read runs/audits/paper-readiness-cleanup-20260910. Historical UI islands and
  completed launchers were removed after external source archival; raw data,
  frozen models/commands/forecasts and independent shared backends are preserved.
  Source-only export includes planning/PVA and an empty model/flight catalog.
  HANDOFF and paper guide now describe the actual selected M0; older scope
  bullets below are historical unless repeated there. No real M1/M2, new planner,
  model/physics change, flight or physical adaptation claim follows from this audit.

- Latest model evolution/evaluation UI is implemented. Read docs/SIM_REAL_EVALUATION.md
  and runs/audits/model-evolution-20260910. Flight comparison has an M0/M1/M2
  workspace; config/evaluation/campaign.json registers only selected M0.
  tools/evaluate_models.py explicitly registers fits/reports or evaluates all
  registered scalar-damping models on identical prepared flight inputs. Viewing
  does not launch work. Training ancestry follows measured recording hashes;
  identical commands on repeated flights are not training leakage. Outcome
  reviews and catalog revisions are preserved. New fitting result hashes bind
  candidates/diagnostics; generation indices propagate beyond M1. New setup
  binds its selected package, not a hard-coded M0 audit. Coverage excludes
  initialization outside the scored interval. No real M1/M2, planner or flight
  ran. Synthetic CUDA/UI fixtures remain audit-only. Preserve current flight
  CSV/forecast and reviewed raw fitting gates; no automatic fit or promotion.

- Latest request authorized careful literature/code audit and preparation for
  M0-flight adaptation. Read docs/M0_TO_M1_ADAPTATION.md and audit
  runs/audits/M0-to-M1-readiness-20260910. `tools/adapt_whip.py` is the new raw
  workflow; old five-take normalized/neural fitter is historical. New inbox
  rehearsal_csv_and_result_in_real_flight/M0_whip_20260910-022818-648386/flight_take
  is empty. Takes whip_001/002 predeclared adaptation, whip_003 validation.
  No real M1 fit, new planner or flight ran; synthetic test candidates are not M1
  evidence. Preserve selected CSV/forecast. Compare original forecast first,
  review clocks/contact/masks/roles, prepare causal 1 s cable / 0.4 s drone state,
  diagnose conditional versus command-driven mismatch, then justify any scalar
  cable damping fit. Drone/NN/EI/Cb unchanged. No automatic fit when data arrives
  or automatic candidate promotion; prospective M1 flight remains necessary.

- Latest user selected MPPI `20260910-022818-648386` and its exact rehearsal
  for the next real-flight take. Read `runs/flight_packages/20260910-022818-648386/README.md`
  and `config/pva/flight_selection.json`. CSV/forecast frozen and checksummed;
  preserve them for prospective measurement before M1 fitting. Eleven other
  MPPI runs/three rehearsals are archived in place (ARCHIVED markers); no
  deletion/moves of original evidence. Do not restore or restart automatically.
  Audit `runs/audits/mppi-live-fix-and-flight-selection-20260910` verifies 3,786
  retired files and 26 selected frozen files. Live UI run-binding bug fixed;
  eight UI tests/native VTK playback checked. No flight/controller/model changes.

- Latest user authorized one new MPPI run with tip-contact success:
  `20260910-022818-648386`, audit `runs/audits/mppi-tip-contact-20260910`.
  Same development M0, 512 samples / complete 1.5 s, reward/search/baselines
  and bounds; only the success criterion differs from the last timing trial.
  Completed: 20 updates / 64.48 s, modeled tip hit 1.124878134 s, retained previous
  sweep's first 34 commands with no improvement. Rehearsal and full recovery
  checked, frozen and selected. No duplicate/restart, automatic second trial,
  fitting or flight. Read HANDOFF and audit review.json for evidence.

- Latest user authorized and completed a simpler MPPI success gate: active
  `task.success_criterion=tip_contact_v1`, feasible tip entry into the existing
  5 cm virtual sphere. Speed/reversal/wave diagnostics no longer veto a hit.
  Preferred-fold objective and weights remain; termination can change scores.
  Missing versions retain legacy semantics. Preserve historical flags/forecasts
  and PPO configuration. Read docs/WHIP_SUCCESS_CONDITION_REVIEW.md and audit
  runs/audits/tip-contact-gate-20260910. Two saved motions pass bounded GPU
  checks; miss/invalid controls fail. No new optimization, fitting, flight or
  replay selection; no automatic follow-up. Older pause notes are historical.

- Latest user request paused further implementation/optimization for a careful
  paper and success-condition review. Read docs/WHIP_SUCCESS_CONDITION_REVIEW.md
  and latest HANDOFF. Review completed without code/settings changes or physics
  rollouts. Both recent motions made modeled tip contact; custom composite
  success remained false. Existing 4 m/s and bend-band/dwell gates are unvalidated
  project choices, not paper-established success requirements. Keep contact,
  feasible execution and preferred-wave evidence separate. No automatic reward
  simplification, detector, MPPI run or fitting campaign; preserve old results.

- Latest user-authorized timing/strength MPPI trial completed on 10 September:
  docs/MPPI_TIMING_SEARCH.md and runs/audits/mppi-timed-baselines-20260910.
  Job 20260910-020448-013542 retained the exact previous new-M0 sweep baseline
  after 20 updates / 64.40 s; no objective improvement, strong fold or strict
  success. 71.25% of update candidates failed existing modeled feasibility
  checks; specific constraint attribution is not established. Rehearsal/recovery
  frozen and selected; prior assets preserved. Same development M0 and reward,
  two exact command baselines, editable timing/strength, 512 samples / 1.5 s.
  All optimization, fitting and flights stopped; no automatic follow-up. Read
  latest HANDOFF for the sequence; older scope bullets below are historical.

- Latest authorized investigation completed on 10 September. Read
  docs/PRELIMINARY1_DAMPING_RESOLUTION.md. Staged UNSELECTED M0 development model
  is runs/audits/preliminary1-gradient-resolution/candidate/model.json: curvature
  smoothing 2e-5, scalar cable damping 0.4/s, cable NN disabled, original drone
  fit inherited. Checked long gradients now agree; old M0/defaults unchanged.
  Representative difficult figure-eight retains substantial error. No neural
  training, MPPI restart or flight. User emphasizes sim-to-real-to-sim and will
  collect new whip takes later: freeze forecast -> measured check -> reviewed
  data for M1 -> next prospective check. This candidate is still M0 development.

- Latest user authorization completed a small cable-only physical fitting pilot:
  docs/PRELIMINARY1_CABLE_PILOT.md and runs/audits/preliminary1-cable-only-pilot.
  Weighted one-second initialization explains most of the pilot improvement;
  stiffness remains weakly determined. Search stopped by practical plateau at
  generation 6; no new M0 was published/selected. MPPI remains stopped, and no
  residual/drone training ran. Preserve pilot evidence and existing M0. The
  full-window gradient issue remains unresolved; no automatic follow-up campaign.

- Earlier user request paused implementation/refitting for a comparison with the
  literature. Read docs/PRELIMINARY1_METHOD_COMPARISON.md. MPPI run
  20260909-224132-314704 is stopped at 40 commands; no replacement fit started.
  Weighted one-second initialization and gradient diagnostics were completed
  before the pause. Long-window gradient agreement remains unresolved. Preserve
  the existing M0; do not automatically resume fitting or MPPI from older notes.

- Latest fitting preparation request: use 1 s causal cable history consistently
  for both cable-window positions (`cable_history_s=1.0` in new protocols).
  Existing frozen fits keep their original semantics; drone history remains 0.4 s.
  No refit was authorized by this history change. Same-window fixed-M0 checks
  were mixed and worsened held-out tip RMS (30.82â†’77.16 cm). Do not claim an
  improvement. See `docs/PRELIMINARY1_FIT_AUDIT.md` and
  `runs/audits/preliminary1-history-1s` before further fitting work.

- Latest user clarification supersedes the 512-sample sequence described below:
  restore old successful MPPI search settings while using preliminary M0.
  Active job `20260909-224132-314704`, 1,024 samples / 2 s lookahead, initial
  minimum/patience 20/12, later 5/4, original reward/noise/task/limits and exact
  editable initial proposal from archived parent `20260909-172921-966356`.
  Start [0,0,1.255], target [1.25,0,1.0]. Every candidate uses new M0; no historical
  accepted forecast/trajectory is substituted. Live 3D enabled. Read status before
  acting; no duplicate/restart. The 512-sample supervisor is stopped/cancelled,
  including its fallback. Audit: `runs/audits/mppi-M0-original-settings`.

- Fresh unseen repaired system, user-reported drone mass 145 g and cable assembly
  mass 17 g (162 g total). Other hardware geometry unchanged. Cable mass is
  provisionally distributed in the previous proportions; individual masses were
  not newly measured. Preserve tracking-origin/attachment offset and all lengths.
- Five new preliminary1 pairs are now imported for review after the clean reset.
  Read HANDOFF and runs/data_review/preliminary1-intake/README.md for pairing,
  user-confirmed hand trimming/no contact, and missing-marker intervals. Controller
  XYZ is OptiTrack-derived (~10 Hz cached updates); native pose is 100 Hz.
  User explicitly authorized naming fixes, careful preparation and GPU fitting.
  Completed job: runs/adaptation/20260909-preliminary1-M0-v2. Read its status before
  acting; no duplicate or automatic campaign. See docs/PRELIMINARY1_FIT.md.
  Raw global XYZ is retained; figure8_002 is held out of fitting/selection.
- User requested a clean workspace before these new recordings.
  Planner for this check is MPPI. No old data/model/policy/replay is selected.
  All old research artifacts were moved to the external archive in HANDOFF,
  with every file SHA-256 verified. Do not restore or restart them automatically.
- The authorized preliminary M0 fit is complete. Candidate remains provisional,
  now user-selected for MPPI: held-out representative coupled tip RMS is 64.6 cm at 2 s.
  Do not describe this as precise strike/flight validation or restart fitting automatically.
  PPO, fitting, heartbeats and flights remain stopped. User-authorized MPPI
  sequence is 512 samples / 1 s (`20260909-223523-507672`), then only if no valid
  modeled strike 512 / 1.5 s (`20260909-223523-778921`). Supervisor/status lives in
  `runs/audits/mppi-512-1s-then-1p5s`; read it before acting. No duplicate/restart.
  Both start from the same archived editable whip seed under preliminary M0.
  Manual stop cancels the remaining sequence. No third trial or broader sweep.
  The prior 256-sample 1 s run `20260909-222603-359485` is stopped at 70 commands.
  The prior quick continuation `20260909-221253-764801` is stopped at 59 commands.
  The 2 s seeded trial `20260909-222142-694207` is stopped with zero committed actions.
  No verified ROS flight sender exists. Do not switch controllers.
- config/experiment.json is the active experiment identity. config/current_vehicle.json
  and both structural model templates carry the new masses. Templates contain
  unfitted engineering initialization, no learned residual or selected checkpoint.
- Keep MPPI's user-authorized quick settings: 512 samples, 1 s lookahead first
  with only the explicit conditional 1.5 s fallback above, one
  30 Hz committed jerk action, 5 s maneuver limit. Rewards and strike criteria remain unchanged. Current PVA start is
  [0,0,1.255] m and target [1.25,0,1.0] m, translating the liked old farther-target
  geometry by +2 m in X. Initial minimum/patience 8/4, later 3/2, no iteration
  ceiling/planning deadline; live candidate 3D enabled. No new hard limits or
  rescaling of provisional flight bounds. Historical seed is 39 exact jerk
  commands: first 30 initialize the horizon, remaining nine enter as editable
  tail guesses as it advances, then zero jerk in the 1 s trial. The 1.5 s fallback
  sees all 39 commands plus six zero-jerk guesses. No old M1 model or forecast
  imported. Provenance is in each run and `runs/audits/mppi-512-1s-then-1p5s`.
  User prefers a maneuver under 2 s, but only the lookahead was changed; do not
  equate lookahead with episode duration or introduce an unrequested hard limit.
- UI remains six pages. Recordings now starts with Preliminary takes. MPPI/PPO
  require a newly selected fitted model; the old hard-coded adp0 fit is not a
  valid new-system input. PPO checkpoint rehearsal functionality remains available.
- Planned evidence chain: new preliminary data -> fresh M0 -> MPPI/frozen forecast
  -> prospective measured flight -> M1 fit from reviewed flight data -> new MPPI
  -> prospective check. No improvement or physical validation has been established.

## Preserve research meaning and artifacts

- Preserve raw logs, failed trials, masks, models, checkpoints, optimizer state,
  source snapshots, flown CSV bytes and their exact saved prediction ghosts.
  Do not regenerate an old ghost with a newer model and call it the original forecast.
- Measured geometry is tracked-origin/attachment-specific. Apply the rotated
  offset consistently; never assume tracked origin equals COM or firmware origin.
- Hover-normalized Z uses retrospective pre/post information. State that limitation
  in paper claims; it is not physical flight validation or measured coordinate calibration.
- Keep cf7 and cf3 data/model identities separate. M0/M1 names require exact provenance.
  All-data development fits are not independent tests. Preserve old split snapshots.
- No change to a provisional simulation bound establishes a measured vehicle limit.
  No verified ROS flight sender was added by the PVA implementation. Use actual
  firmware/interface details rather than inventing topics, frames or parameters.
- Latest user clarification: the old sudden drop was caused by acceleration demands
  exceeding drone capability. Battery effects on other tracking errors remain unconfirmed.
  Bolt battery is 2S; PVA-based battery calibration is proposed, not implemented.
- Read `docs/FUTURE_ADAPTATION_FITTING.md` before any newly authorized fit.
  Use practical plateau stopping, preserve best weights/state, and distinguish
  convergence from manual stops or safety ceilings. No automatic fold/ablation campaigns.
- Historical selected flight policy: `runs/ppo/20260908-195207-486249-seed655/checkpoints/best_validation.pt`,
  SHA256 `d10657f471deb8b22cafbee9008792e8378c8c764ca97039d6dbe7afcc260eea`.
  Later separate study outputs do not rewrite which policy produced earlier flights.
- Preserve independent legacy CEM/force/Isaac assets. PhysX work stays paused.
- Do not reset the working tree, rewrite Git history, or permanently remove
  previously protected research artifacts as cleanup. Retired external archives
  remain recoverable through their manifests; do not recreate or delete them.
- Prefer targeted existing tests. Report the actual tested hardware and OS;
  current PVA verification is Windows/RTX 4080, not Ubuntu/other-GPU validation.

## Documentation

`docs/README.md` is the active reading list. Superseded test reports, run diaries,
old UI instructions and prior versions of these instructions are under
`docs/history/20260909-doc-cleanup/`. They describe historical decisions and
must not be treated as live run instructions. Keep necessary evidence accessible
through links rather than restoring obsolete progress notes to the active index.
