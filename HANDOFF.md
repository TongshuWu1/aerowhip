# Current project handoff

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


Updated 10 September 2026. Branch: `twin-rewrite`.

## Current decision

Latest user requested a detailed M0/M1/M2 system/literature/paper comparison and
clearer UI. Completed: read `docs/M0_M1_M2_SYSTEM_COMPARISON.md` and
`runs/evaluation/M0-M1-M2-system-review-20260910-v2`. Canonical flown lineage is
M0 â†’ M1-full â†’ M2-frozen-refit-v1; gain-only M1 is a rejected sibling and original
M2/refit are the same model signature. New frozen-model diagnostics on three M2
flights (none trained on them) give complete tip RMS 14.86/7.99/7.03 cm; M2 worsens
take 002 despite its lower mean. Prospective own-forecast tip means 17.17/8.77/9.38
cm and observed 5 cm entries 0/5, 0/5, 0/3: model improvement is not yet reliable
real targeting. All current data remains development. No fit/planner/controller/
flight selection change. 36 fixed-model complete take predictions plus conditional
diagnostics took 139.27 s on Windows/RTX 4080; prior M1/M2 metrics reproduced
exactly, 231 originals verified unchanged. Flight comparison â†’ Study overview
separates original flights from matched-model diagnostics, excludes training
ancestry by default, links original ghosts and this report. Existing variants,
traces and fitting UI remain. 19 tests and native Qt/VTK replay/layout checks pass.
The first preparation failed before simulation on empty original M1 role mapping;
corrected audit uses reviewed later roles, preserving both attempts. Initial-state
robustness, combined fitting extensions and clean prospective paper study remain
separate next work; do not automatically fit or replan from this review.

Latest discussion: user proposes PPO as a fast initial-state-conditioned offline
planner, then executes its generated PVA open-loop (not live PPO feedback).
Requested latency benchmark completed; read
`runs/audits/ppo-state-to-pva-latency-20260910/README.md`. Existing M0 PPO, measured
M2 take 001 initial state: warm whip PVA 1.295 s, causal state estimate 81 ms,
recovery construction 96 ms, about 1.48 s total command construction. Full checked
recovery forecast totals about 10.74 s. One network action is 0.149 ms. RTX 4080 /
Windows, no training or production changes. Same old policy in M2 fails at 0.9 s,
so its 0.993 s timing is an invalid prefix, not a complete M2 plan. All selections,
models/raw data/original ghosts preserved; benchmark arrays are audit-only. No
M2 PPO training or physical flight is authorized by this timing request.

Three real M2 flights are now reviewed. Read
`runs/data_review/M2-whip-intake-20260910/README.md` and
`runs/data_review/M2-whip-first-comparison/performance.json`. User confirms complete
three-take batch, unchanged CSV/hardware/controller/origin, no contact/intervention.
001/002 retain adaptation roles, 003 validation; 004/005 will not follow. All 310
logged CSV rows match exactly. Original-forecast mean drone/tip RMS is 7.04/9.38 cm.
Observed virtual-target entries 0/3; closest centre distances 8.63/6.56/8.84 cm,
forward near-approach speeds 5.76/4.54/5.92 m/s. Clocks remain estimated, raw XYZ
and gaps preserved. M2-frozen-refit-v1 Flights by model now has all three original
ghost replays. Nine tests and native Qt/VTK playback pass. No new fit/rollout,
planner, promotion or flight command; prior selections/models/forecasts unchanged.
Frozen selection-preparation protocol's no-flight field describes its creation
time; actual new collection facts are in batch `collection_context.json`.

Latest user explicitly selected the screenshot's rehearsal
`20260910-211435-608306-M2-frozen-refit-v1-whip` for use. Exact flight selection
is now in `config/pva/flight_selection.json`; former M0 pointer is archived under
`config/pva/flight_selection_history/` and in `runs/audits/M2-flight-selection-20260910`.
Frozen package: `runs/flight_packages/20260910-211435-608306`. Full CSV is 10.3 s
including recovery; command hash b34ddfe74a0292905c2173082f4d2a76e0b798c8c2e28dc8b7ce1be9b9c2d6b3,
forecast hash 452300b7067cc5cd324dde2954d231ed15ff6ebbcb49911b5fc9f80a321a7656.
Existing user-exported CSV matched exactly and was preserved. Empty inbox:
`rehearsal_csv_and_result_in_real_flight/M2_whip_20260910-211435-608306/flight_take`.
Raw protocol binds this exact M2 forecast/model and predeclares whip_m2_001/002/004
adaptation, 003/005 validation. Full staged method is planned after data review and
explicit fit authorization; legacy scalar-only setup defaults do not apply.
2,399 protected files unchanged; only the authorized selection pointer changed.
No model fitting, new planner or physical flight was started. Selection for a
prospective test does not claim model-validation promotion. Preserve all prior
CSV/ghost/model evidence; compare incoming measurements with this original ghost.

Latest user requests another hitting-velocity reward increase. Completed
job `runs/mppi_pva/20260910-211435-608306`, audit `runs/audits/M2-impact1600-20260910`.
20 updates / 67.93 s, plateau. Exact prior 35 commands, complete CSV and forecast
retained; speed remains 4.90513 m/s. No improvement beyond the changed bonus.
Independent replay/full recovery/native rendered frames pass; 2,372 originals
unchanged. Separate rehearsal open paused at start. See audit README/review.json.
Read status; no duplicate/restart. Impact weight 800â†’1600; same M2 refit, 512/1.5 s,
all other numerical settings and original M1/M0 baseline/reference files unchanged.
Previous M2 winner is an editable initial incumbent, not a substituted forecast.
CUDA preflight reproduces 4.90513 m/s and rescored objective 1892.43069. Four impact
tests pass. See docs/M2_AGGRESSIVE_MPPI.md. Separate live view/rehearsal/recovery;
preserve earlier runs and all model/flight/global selections; no new fitting or flight.

Latest user explicitly requests MPPI with the refitted M2 and a more aggressive,
harder hit. Authorized job `runs/mppi_pva/20260910-210545-160853`; audit
`runs/audits/M2-aggressive-mppi-20260910`; read status before acting, no duplicate.
Completed: 26 updates / 85.11 s, plateau. Independent modeled contact at
1.152791 s; 35-command whip (1.166667 s), 10.3 s complete CSV/recovery.
Forward hit speed 4.50946â†’4.90513 m/s against the same M2 model's M1-command
baseline (+8.77%; squared-speed proxy +18.32%, not measured impact power).
Rehearsal `20260910-210545-160853-M2-frozen-refit-v1-whip` opens paused at start.
Independent score/contact-speed/full recovery and native live/replay checks pass;
1,957 protected files unchanged. Result audit README/review.json. No restart.
See `docs/M2_AGGRESSIVE_MPPI.md`. Uses M2-frozen-refit-v1, 512 samples / 1.5 s,
timed editable M1/M0 flown command baselines, successful-contact speed bonus 800
(prepared recipe was 400), scale 4 m/s, contact-first ranking, same shape objective
and bounds. Sixteen tests and M2 baseline replay pass. M1 commands rerolled in M2
predict a hit at 4.50946 m/s; use that same-model speed comparison. Live 3D and
independent rehearsal/recovery enabled. Preserve prior data/models/flights/config;
no fitting or automatic promotion/physical flight. This explicit authorization
supersedes the no-new-MPPI scope of the completed refit below.

Latest authorization: document/freeze the existing staged regularized simulation-error
identification method and run one separate M2 refit. Read
`docs/FROZEN_SYSTEM_IDENTIFICATION.md` and `runs/audits/M2-frozen-refit-v1`.
Job `runs/adaptation/M2-frozen-refit-v1` starts from exact M1-full with the same
data/numerical contract as the earlier M2, except for the new candidate ID.
Completed in 1,656.6 s (27.61 min), Windows/RTX 4080. Do not duplicate/restart.
Nominal parameters, both NN tensors, training-loss histories, candidate signature
and every reported per-take RMS reproduce original M2 exactly. Both residual
gradient probes and runtime parity passed; 739 protected files are unchanged.
Native Qt comparison/fit-progress check passed (6/6 stages). All-five combined
tip RMS remains mixed: M1 5.846 cm, original M2/refit 6.232 cm; held-out mean
6.820â†’6.421 cm hides opposite changes on 003/005. Registered as an unpromoted
generation-2 variant. Result: `runs/audits/M2-frozen-refit-v1/README.md`.
This is a reproducibility refit, not implementation of the proposed combined
fitting objective below. Old M2,
raw recordings, selected flights and planner settings remain preserved. No
automatic promotion, MPPI or real flight is authorized by this refit.

Latest user clarification: the paper is about the **whole UAV open-loop dynamic
whipping system**, combining real-to-sim-to-real model updates with MPPI. They
explicitly do NOT claim a new adaptation algorithm. The immediate task is to settle
a repeatable supporting identification procedure. Use regularized weighted refitting
from the parent, retaining eligible training data, with staged component fitting
and proposed final combined refinement. See docs/SYSTEMATIC_ADAPTATION_PROTOCOL.md
v0.2. Do not expand scope into proving warm-start adaptation beats all-data refitting
or inventing a novel adaptation method. Such comparisons are optional for specific
system claims. Current runs are development; clean collection follows method freeze.
Documentation framing corrected; no implementation, fit, planner or promotion ran.

The user requests a systematic adaptation method for the paper and explicitly
states that all current runs are development only. They will collect a separate
clean experiment after the method is refined. Read
`docs/SYSTEMATIC_ADAPTATION_PROTOCOL.md` (design v0.1, not a frozen or implemented
new fitter). It specifies one regularized drone/conditional-cable/complete-cable
objective, staged initialization plus combined refinement, fixed replay/validation
rules, and a fresh M0 from new preliminary data for the clean study. Current learned
weights must not silently seed that fresh M0. Freeze the method/planner/evaluation
before clean collection. Current data/artifacts/roles remain unchanged; no cleanup,
fit, planner or promotion was performed. Combined fitting and development comparison
are the proposed next work, not an authorized running job.

The latest user asked why M2's combined prediction is mixed. Frozen-model analysis
is complete: read `docs/M2_REGRESSION_ANALYSIS.md` and
`runs/audits/M2-regression-analysis-20260910`. Separate component losses omit a
training-time command-to-tip objective. Fixed component/XYZ swaps show the changed
forward attachment motion drives most late tip-height regression on 001/004/005,
while helping larger-excursion 003. M1 error cancellation also contributes.
Own-model cable replays match within 1.83e-12 m; 97 protected files are unchanged.
Clock-shift sensitivity preserves those four takes' direction but qualifies the
small mean difference. Final coupled refinement with component retention terms
is proposed, not implemented or run. No new fit/MPPI/promotion/flight occurred.

The user authorized another **full M1-full â†’ M2 adaptation** and superseded the
proposed hard 4 m/s requirement with a stronger soft speed reward. Read
`docs/M1_TO_M2_ADAPTATION.md` and `runs/audits/M2-full-adaptation-20260910`.
Job `runs/adaptation/M2-full-whip-v1` completed in 1,667.3 s (27.8 min) on
Windows/RTX 4080. M2 is registered, not promoted; do not duplicate or restart.
User confirms all five M1 takes used
unchanged controller/hardware/CSV, with no contact or intervention. New takes
001/002/004 train; 003/005 are excluded from fitting/selection but previously
inspected as M1 outcomes. Both residuals warm-start from exact M1-full weights.
Training mass is 50% new M1 takes, 25% old M0 adaptation-only takes and 25%
preliminary training; prior held-out takes remain excluded. Preparation parity,
parent identity, CUDA inherited weights, 42 focused tests and native UI checks pass.
Both selected residuals passed full temporal gradient checks. Drone NN stopped
at 260 updates; cable NN stopped at 200, selected update 175, both by plateau.
Held-out 003/005 mean RMS: drone 9.18â†’6.53 cm, conditional tip 5.74â†’4.58 cm,
combined tip 6.82â†’6.42 cm. Combined tip is mixed: 005 worsens 4.97â†’6.99 cm.
Across all five new takes (including training), drone 8.74â†’5.97 cm and conditional
tip 5.63â†’4.42 cm improve, but combined tip worsens 5.85â†’6.23 cm. Do not claim
uniform end-to-end improvement. Saved-array decomposition shows opposing errors
in M1 and changed cancellation; no validation-driven refit followed. Earlier M0
holdout 005 improves drone/conditional/combined 7.64/3.95/8.81â†’5.78/3.25/8.20 cm.
Preliminary holdout mean-window drone/conditional tip slightly worsen
7.80/4.92â†’8.09/5.04 cm. All checks follow frozen selection. Candidate model:
`runs/adaptation/M2-full-whip-v1/candidate/model.json`, signature
`be4bd82c038dad1e2f1508babc9a1c3da121e1f428f2a375942b382bf6a7b0b9`.
UI opens Compare generations â†’ M2-full-whip-v1, validation takes. Full fitting
progress is 6/6. Original forecast and all 70 protected files remain unchanged.
Future M2 recipe `config/pva/m2_mppi.json` adds 400*vÂ²/(16+vÂ²) at feasible tip
contact, no speed cutoff or earlier-hit reward. Four m/s is only a smooth scale.
Flown M1/M0 artifacts and global planner/PPO/flight selections stay unchanged.
No new MPPI or flight has launched. Older hard-minimum/unconfirmed/no-fit notes
below describe the previous decision and are superseded by this paragraph.

Five **real M1-full flights** are now processed. Read
`runs/data_review/M1-whip-intake-20260910/README.md` and
`runs/data_review/M1-whip-first-comparison/performance.json`. All 311 CSV packets
match exactly in all five pairs. OptiTrack contains cf_3/cf_6; only cf_3 matches
controller-cached measured XYZ. Identity sidecars bind that measured evidence.
Explicit user rigid-body and hardware/controller/contact/intervention confirmations
remain pending; no review acceptance for fitting has been fabricated. Clock
alignment is estimated from measured streams; raw XYZ/gaps remain unchanged.
Same M1 commands / identical nominal initial state: mean drone RMS M0 10.64â†’M1
8.46 cm, tip 18.85â†’8.77 cm, all markers 13.50â†’8.31 cm. Average gains remain under
Â±40 ms clock sensitivity. M0 is a clearly labeled new counterfactual; M1 replay
matches its original forecast exactly. Separate common causal-history diagnostics
give drone 9.35â†’8.74 cm, conditional tip 9.20â†’5.63 cm, coupled tip 15.98â†’5.85 cm.
No fitting, new MPPI, model promotion or historical ghost change occurred.
Measured near-target outward tip speed averages 5.11 m/s (4.59â€“5.41), versus
6.06 m/s in prior M0 flights (different commands). All five remain outside the
unchanged 5 cm virtual sphere; closest distances 5.87â€“8.54 cm including the first
0.5 s recovery. These are geometric observations, not confirmed physical contact.
User requests restoration of a hard 4 m/s outward tip-speed minimum at contact
**for M2 only**. Requirement saved in config/pva/m2_planning_requirements.json;
no M2 exists or runtime gate is applied to M1. Do not reintroduce the entire old
composite wave gate or relabel M1 outcomes. M2 take roles/fitting remain unassigned.
All five M1 takes are registered by actual model signature, not generation integer;
Flight comparison â†’ M1-full opens take 001 with original ghost. Replay startup now
targets this take; global flight/model selection remains M0. 37 tests and native
Windows/RTX 4080 CUDA/Qt/VTK checks pass. All 11 original uploaded CSVs and 80
protected prior files unchanged; previous catalog revisions and 174 source files
preserved. This supersedes older statements that no real M1 flight exists.

The user requested the next MPPI/real-flight step. A new **M1-full prospective
flight candidate** is prepared: `runs/mppi_pva/20260910-181929-716218`, with checked
rehearsal `runs/rehearsals_pva/20260910-181929-716218-M1-full-whip` and convenient
CSV/forecast/ZIP/guide in `runs/flight_packages/20260910-181929-716218`.
Read `runs/audits/M1-full-mppi-contact-20260910/README.md`. First same-settings M1
run `20260910-181640-773710` finished with a 7.09 cm miss despite sampling hits;
it is preserved. The follow-up explicitly prioritizes feasible tip contact, then
the original objective (`tip_contact_then_score_v1`); historical missing fields
retain their original ordering. No reward-weight, model, bound, controller or
PPO changes. Same frozen M1-full with both residuals, 512 samples / complete 1.5 s,
start [0,0,1.255], target [1.25,0,1.0], original editable seeds/reference.
Completed 31 updates / 102.35 s by plateau: modeled hit 1.177714689 s, 36 commands /
1.2 s whip, full CSV 10.333333 s including recovery. Independent CUDA score and
export-prefix parity passed; native Qt/VTK cast/contact/recovery replay checked.
19 focused tests pass. All 53 protected prior model/catalog/config/flight files
remain unchanged; M0 is still the global flight selection, with the new candidate
separate for review. No physical M1 flight, new fit or model promotion occurred.
The app is open at this new rehearsal, paused at the start. Next: user reviews
the motion, flies its exact fullstate CSV through the existing sender, supplies
paired controller/OptiTrack logs; compare the frozen M1 forecast before M2 fitting.
M0-to-M1 full fitting is already complete. Do not duplicate these jobs or call the
prospective result a model-only improvement: planner selection also changed.

The user explicitly authorized the **complete adaptation implementation and run**.
Full continuation `runs/adaptation/M1-full-whip-v2` completed and registered
**M1-full**, without promotion. On excluded whip 005, drone RMS improves
8.93â†’7.64 cm, conditional tip 6.74â†’3.95 cm, and command-driven tip 14.89â†’8.81 cm.
Take 004's command-driven tip worsens 10.40â†’10.98 cm. Preliminary holdout mean
window drone RMS improves 9.81â†’7.80 cm; conditional tip is nearly unchanged,
4.94â†’4.92 cm. These are retrospective, previously inspected development checks,
not a new blind test or physical M1 flight. M0 remains selected.
Residual removals are mixed: drone NN off improves 005 drone/tip to 6.26/8.21 cm
but worsens preliminary drone mean to 12.71 cm (7.80 with NN). Cable NN improves
005 conditional tip 6.58â†’3.95 cm and coupled tip 11.57â†’8.81 cm, with a slight
preliminary regression relative to updated physics alone (4.82â†’4.92 cm).
Keep the frozen full candidate; neither universal NN necessity nor a new winner
selected from the removal diagnostics is established. Both residual bounds are
Â±0.5 m/sÂ² and occasionally active (~6% near-bound sampled training components).
No parameter bound/evaluation ceiling bound the nominal fit. User asks to judge
all limits by evidence, not remove them blindly. No new bound sweep is running.
The v1 attempt stopped before cable NN updates when its final physical iterate
failed a full gradient check. It remains intact. V2 resumes the exact update-400
drone Adam state without a routine neural ceiling and uses the best numerically
verified physical iterate (all three parameters fitted). No data/window/physics
change. Drone NN stopped at 585 total updates and cable NN at 280, by practical
plateau; both selected checkpoints passed numerical verification. V2 took
1950.1 s including validation, excluding prior V1 work and numerical investigation.
Read [window lengths, residual evidence and stopping review](docs/TRAINING_HORIZON_RESIDUAL_REVIEW.md).
Read [full implementation and experiment](docs/FULL_MODEL_ADAPTATION.md).
It fits drone gains/delay/attitude, drone NN, cable EI/Cb/external damping and a
new zero-initialized cable NN, with 001/002/004 plus preliminary training replay.
Selection is frozen before 005/preliminary holdout evaluation. Do not duplicate,
silently narrow, restart or promote it. M0 and the earlier partial M1 remain intact.
The review-only and missing-implementation statements below describe the prior
decision; the full workflow and its measured-data run are complete. No fitting,
new MPPI or flight should start automatically. See the saved full comparison in
`runs/evaluation/M1-full-whip-v2` and audit `runs/audits/full-adaptation-20260910`.
Thirty-eight focused tests passed. Native comparison UI shows M0â†’M1 and M0â†’M1-full
as separate parent edges, with the full report open; component report is also in
the evaluation selector. Original models/data/CSV/forecast/selection remain intact.

The user clarified that adaptation must cover the **drone model, cable model and
their residual corrections**, as intended by the original research proposal.
Read [full-model adaptation contract](docs/ADAPTATION_MODEL_CONTRACT.md) first.
The raw whip workflow currently lacks residual-training stages and a combined
candidate contract; the preliminary fitter has those learning components under a
different input contract. Do not silently narrow full adaptation to one gain or
one cable coefficient. The latest work is a code/design review, not another fit.
Regression evidence is in `runs/audits/M1-regression-audit-20260910`: raw preparation
reconstructs exactly, numerical discrepancies are tiny, but gain-only M1 worsens
attitude throughout and leaves common late vertical error. No model, controller,
command, forecast or data-role changes. Both subsystems and residuals require an
explicit decision in the next complete workflow; unsupported changes need not be
forced. Older scalar-only scope notes describe previous trials, not the project goal.

First **partial, gain-only** real adaptation trial completed, but **M0 remains selected**. Read
[first adaptation review](docs/M0_M1_FIRST_ADAPTATION.md). Initial diagnosis job
`runs/adaptation/M1-whip-first` motivated a new reviewed gain-only job
`runs/adaptation/M1-whip-response-v1`. Only simulated horizontal feedforward response
changed 0.74668â†’0.61834; cable/NN/delay/controller/commands fixed. Five updates,
57.3 s CUDA float64 practical plateau. Adaptation mean drone RMS 10.71â†’8.14 cm,
tip 12.61â†’12.36 cm, but held-out 005 worsens: drone 8.93â†’10.41 cm, tip 14.89â†’17.93 cm.
Candidate M1 is registered for inspection, explicitly not promoted. Do not retune
on 005 and claim independent validation or launch a new planner automatically.
003 remains excluded from strict cable initialization, never training. The original
M0 forecast stays frozen. Comparison: `runs/evaluation/M0-M1-whip-response-v1`.
34 tests, synthetic gain check, independent replay, clock sensitivity and source
hashes passed; evidence `runs/audits/M0-real-adaptation-20260910`. The implemented
response branch is one nominal gain only, not saturation/capability/NN fitting.

Latest UI organization: Flight comparison â†’ Flights by model uses **Model generation
â†’ Plan/session â†’ Take**, with adaptation/validation labels. Recordings shares the
generation/session filter. Compare generations is a separate tab for cross-model
evaluation. Only M0 exists; M1 is explicitly empty. Switching generation clears the
old ghost. Registering further models/flight batches extends the library; Refresh
reloads it. Existing M0 comparison v2 is now registered in campaign.json, preserving
the previous catalog revision. No data/model/forecast/selection change, fit or rollout.
27 focused tests and native Windows/RTX 4080 M0â†’M1â†’M0 replay checked. Audit:
`runs/audits/flight-generations-ui-20260910`. App left open at M0 take 001.

M0 paired ghost replay is available at Flight comparison â†’ Real flight + plan ghost.
All five takes are listed; startup selection in `config/pva/flight_replay.json` opens
001 at quarter speed, Side XZ. Orange measured motion and translucent blue original
MPPI execution share one timeline. Shortcut now loads its take; automatic ghost
selection binds to the batch protocol and forecast checksum. Original flight/replay
selection files and research evidence unchanged. Fourteen focused tests and native
Windows/RTX 4080 VTK playback passed. See `runs/audits/M0-flight-ghost-ui-20260910`.
No fitting/planning/flight ran. App left open, paused for user playback.

M0 real flight data have arrived: five `whip_m0_001`â€“`005` paired takes in the
prepared flight_take inbox. User confirms same CSV/settings, no contact, and
requests three adaptation takes. Use 001/002/004 adaptation, 003/005 validation.
Read `runs/data_review/M0-whip-intake-20260910/README.md`. All 309 logged CSV packets
match in every take; raw files and frozen M0 verified unchanged. Estimated measured-
stream clock alignment is saved (not independent synchronization). Current frozen
comparison: `runs/data_review/M0-whip-first-comparison-v2`; v1 predates split revision.
003 has missing c5 history and cannot pass current strict cable initialization;
retain as validation evidence, never silently impute or move into training. 005
has usable history. No prepared fit job, diagnostic rollout or M1 fit has run.
Older empty-inbox statements below are superseded by this intake.

Latest user chose to keep the original selected MPPI and latest PPO only in the
active libraries. MPPI `20260910-022818-648386` and its M0-development-whip rehearsal
remain selected for flight. PPO run `20260910-115619-908807` and its latest saved
preview `20260910-173200-563888-ppo-preview` remain visible; that preview snapshots
`latest.pt` at 491,520 attempts and is preview-only (modeled miss, not a flight export).
Seven duplicate/older entries, including the harder-hit rerun, are archived in
place with ARCHIVED markers. All 5,970 existing files checked unchanged; no files
deleted, rollouts launched, or flight/config selection changed. Audit:
`runs/audits/keep-latest-ppo-selected-mppi-20260910/result.json`.
Future harder-hit reward settings remain as below; this selection did not revert them.

Latest correction: user meant harder impact, not earlier hit time. Earlier-hit
job 20260910-135950-415845 was cancelled before launch. Future MPPI defaults
replace early-hit fields with impact=200 and impact_scale_m_s=4: successful
directed tip speed v earns 200*vÂ²/(16+vÂ²), no explicit time bonus or new cap.
Corrected one-run job is 20260910-140256-116887, same M0/512 samples/1.5 s/search.
Completed: 20 updates / 64.87 s, retained the exact original 34-command motion;
directed impact speed 4.17886 m/s, modeled hit 1.124878 s. Score 799.5256 includes
104.3717 new bonus; no motion improvement. Separate rehearsal/recovery passed and
full CSV matches the original selected bytes. Original flight selection unchanged.
Read runs/audits/mppi-harder-hit-20260910/status.json before acting; no duplicate.
Original M0 flight package/selection and PPO remain unchanged. A new rehearsal
is saved separately after the run; no automatic flight selection or execution.
Read docs/MPPI_PREFERRED_FOLD.md. The earlier-hit notes below are historical.

Latest user requested more reward for faster MPPI tip hits. New future MPPI
settings use reward.early_hit=100 and early_hit_scale_s=1: a one-time bonus
100*exp(-actual_hit_time/1s), only for successful feasible outcomes. Missing
weights retain historical zero; the timescale is independent of lookahead and
episode duration. Both streaming returns and whole-trajectory/preferred-fold
scores use the bonus. PPO settings and selected flight artifacts are unchanged.
23 tests plus Windows/RTX 4080 fixed-command scoring checks passed; no new
optimization/flight ran. Audit: runs/audits/mppi-earlier-hit-20260910.

Latest user authorized literature/math review and adaptation preparation while
collecting M0 flight data. Read [drone-response adaptation](docs/DRONE_RESPONSE_ADAPTATION.md)
and `runs/audits/drone-response-readiness-20260910`. New `adapt_whip.py diagnose-drone`
uses adaptation takes only, native-pose derivative diagnostics and seven production
pose replays for local gain/delay/attitude-lag sensitivity. No drone candidate or
cap is fitted/published; the reviewed scalar cable fitter remains unchanged.
37 focused/source-release tests passed and a fresh Windows/RTX 4080 synthetic
pipeline/math check passed. No real data was fitted, no new planner/flight ran,
selected/retired evidence hashes unchanged. Choose any drone-update family from
reviewed incoming data; do not mistake an observed acceleration maximum or local
sensitivity rank for a physical actuator limit. Actual sender/logger still unverified.

Latest integrated audit: [M0â†’M1 procedure](docs/M0_TO_M1_ADAPTATION.md).
User clarified that capability should be learned as command-to-executed-motion
response, not a new hard generated-acceleration cap. No limit was added and the
selected M0/CSV remain unchanged. Training-range extrapolation alone does not
veto an experiment; actual sender/logger semantics and controller identity remain
unverified. Do not confuse the training maximum with a measured vehicle limit. Diagnose on adaptation
takes only; freeze the candidate before parent/candidate validation diagnostics.
The scalar cable fitter now enforces this diagnostic ordering. A small drone-response
update is a proposed branch requiring a new reviewed protocol, not implemented
drone/saturation fitting. Preserve the selected package and empty real-data inbox.

The selected development-M0 MPPI flight command and original forecast are frozen
for prospective measurement. The separate user-authorized PPO trial has completed;
fitting, MPPI and flight execution remain stopped. No real M1/M2 or physical adaptation improvement exists. Do not restart
old jobs or restore retired artifacts automatically.

Previous request: make PPO learn with the same development M0 and actual objective
as the selected MPPI run. Read [PPO objective alignment](docs/PPO_MPPI_OBJECTIVE.md).
Job `runs/ppo_pva/20260910-115619-908807` trained from scratch, 2,048 CUDA environments,
1.5 s, exact selected MPPI start/target, preferred-fold objective and tip-contact
success. It does not imitate MPPI commands. Read its live status before acting;
do not duplicate or automatically restart. The selected flight package is unchanged.
This supersedes the historical PPO stop notes only. Audit and score-parity evidence:
`runs/audits/ppo-mppi-objective-20260910`.

Latest experiment audit: [drone command chain](docs/DRONE_COMMAND_CHAIN_AUDIT.md).
Raw extraction, model asset lineage, jerkâ†’CSV and exact commandâ†’forecast checks
passed; selected flight bytes unchanged. Main physical-prediction concern:
selected command acceleration 14.31 m/sÂ² versus 8.18 m/sÂ² represented in drone
training. Actual flight sender/logger still needs identification; logger timestamps
are not onboard acknowledgements. One-way loaded-model coupling and ideal hover
initialization remain explicit limits. No fit/model/command/controller change.

PPO completed normally at 491,520 attempts with reward-plateau stopping; best
deterministic fixed-scenario score 692.8127, modeled contact true. This status
was read during the command-chain audit; no PPO restart or new final-policy
rehearsal was run. Earlier frozen rehearsal checks below keep their own scores.

Independently frozen/replayed PPO checkpoint at 139,264 attempts scored 560.3135,
made modeled tip contact and passed complete recovery. Saved separately at
`runs/rehearsals_pva/20260910-120514-716759-ppo-matched-objective`; not selected
for flight. MPPI scored 695.1539; PPO's fold component is still substantially
lower. Training continued beyond the frozen review; consult status rather than
assuming convergence or real-flight validation. 59 focused tests passed, one
historical-fixture skip; 26 selected/6 package/3,786 retired evidence hashes intact.

Previous task: theory/implementation audit and removal of unused legacy source.
Read [paper readiness](docs/PAPER_READINESS_REVIEW.md) and the rewritten
[paper handoff](docs/PAPER_WRITING_HANDOFF.md). The method is MPPI-inspired offline
trajectory optimization plus a loaded-drone/cable forward predictor and proposed
between-trial scalar adaptation. It is not exact MPPI theory, online MPC, or a
physically validated sim-real adaptation result. Verification and exact source
removals are in `runs/audits/paper-readiness-cleanup-20260910`.

The current UI keeps six pages. Unused historical pages and the older desktop
shell were removed. Model fitting cannot launch the retired normalized bootstrap.
Source-only export now includes planning/PVA and starts with no selected model,
flight or fit. Shared numerical/back-end compatibility modules remain intentionally.

## Selected experiment

- Aircraft/cable: 145 g / 17 g, 2S battery. Cable mass distribution was scaled
  proportionally; geometry and tracking-origin offset remain unchanged.
- User reports Mellinger now works with fivefold integral gain. Do not switch
  controllers. Prior sudden drop was attributed to excessive requested acceleration.
- Start tracked origin `[0,0,1.255]` m; target `[1.25,0,1.0]` m; 5 cm sphere.
- Development M0: `runs/audits/preliminary1-gradient-resolution/candidate/model.json`.
  Scalar cable damping 0.4/s, curvature regularization 2e-5, cable NN disabled;
  original preliminary drone fit inherited. EI/Cb remain weakly identified.
- Selected job: `runs/mppi_pva/20260910-022818-648386`.
- Rehearsal: `runs/rehearsals_pva/20260910-022818-648386-M0-development-whip`.
- Package: `runs/flight_packages/20260910-022818-648386`; instructions in its README.
- Selection: `config/pva/flight_selection.json`. Preserve this exact selection.

CSV SHA-256: `ea2e20bba92eb55cad12559ea97fb817eed76cd9ef6b21c191bf7b504368dde3`.
Original forecast SHA-256: `8ed2d231f333d567d71fd8871322e6a2d77b625a29ea2135c434553290b25e13`.

Selected search: 512 samples / complete 1.5 s, 20 updates / 64.48 s; retained
the prior new-M0 baseline without objective improvement. Exported whip has
34 commands / 1.1333 s, modeled tip contact 1.124878134 s, total recovery/hold
10.2667 s. Settled initial hover, takeoff and landing are outside the CSV.
No verified ROS flight sender was added. Active MPPI uses `tip_contact_v1`;
historical flags and PPO keep their own original semantics. Fold/style preferences
do not establish measured energy transfer or gate current tip-contact success.

## Next measured-data loop

Read [M0â†’M1 adaptation](docs/M0_TO_M1_ADAPTATION.md) and
[model-generation evaluation](docs/SIM_REAL_EVALUATION.md).
Inbox: `rehearsal_csv_and_result_in_real_flight/M0_whip_20260910-022818-648386/flight_take`.
It is empty. Predeclared takes: whip_001/002 adaptation; whip_003 validation.

Compare the original forecast first. Review clocks, exact commands, raw masks,
contact/free-motion intervals and take roles. Use raw coordinates and causal
1 s cable / 0.4 s drone history. Distinguish cable error conditional on measured
attachment from command-driven error. Only justify a scalar cable damping update
after diagnosis; drone/NN/EI/Cb/geometry/regularization remain fixed. No automatic
fit when data arrives or candidate promotion. A prospective M1 flight is required.

`tools/adapt_whip.py` implements setup/compare/prepare/diagnose/fit.
`tools/evaluate_models.py` manages lineage and common-flight comparisons.
`config/evaluation/campaign.json` registers only real M0. Same-flight evaluation
currently accepts the scalar-damping lineage; changed drone/geometry/residuals
require a new reviewed protocol. Synthetic M1/M2 are software tests only, in
`runs/audits/model-evolution-20260910`. Prepared jobs bind source hashes: after a
code change prepare a new job; do not modify frozen code snapshots to bypass checks.

## Preliminary evidence and archives

Five reviewed preliminary pairs are retained; controller XYZ is cached
OptiTrack-derived position, not an independent sensor. Native tracking is 100 Hz.
The user confirmed hand trimming and no contact/intervention/settings changes.
Preserve missing markers and native raw coordinates. `figure8_002` was held out
of fitting but inspected during development, so is not a blind final test.
Read `docs/PRELIMINARY1_FIT.md`, `docs/PRELIMINARY1_FIT_AUDIT.md`,
`docs/PRELIMINARY1_CABLE_PILOT.md` and `docs/PRELIMINARY1_DAMPING_RESOLUTION.md`.
Difficult-motion errors remain substantial; causal initialization did not
uniformly improve the original fit.

Eleven old MPPI runs and three rehearsals remain archived **in place**, not
deleted. The 3,786-file manifest is in
`runs/audits/mppi-live-fix-and-flight-selection-20260910`.
The earlier hardware reset archive is outside the repository at
`C:\Users\wts28\Documents\PHD\particle_filter_cable_project_archive_20260909_unseen_145g_17g`;
its `manifest.json` records original paths and hashes.

The pre-cleanup source/docs/config snapshot is outside the repo at
`C:\Users\wts28\Documents\PHD\particle_filter_cable_project_legacy_code_20260910\source_before_cleanup.zip`.
All 615 entries were hash-verified before source removal. Its manifest and archive
hash are recorded in the current audit. This also preserves the long superseded
handoff and earlier paper notes. Never delete raw logs, checkpoints, failed trials,
optimizer state, flown CSV bytes or original forecasts as source cleanup.

Current source-only verification uses the existing Windows Python environment;
GPU checks use RTX 4080/float64. Do not claim fresh installation or other-platform
validation. No new real fitting campaign, planner optimization or flight was
authorized by this audit.
