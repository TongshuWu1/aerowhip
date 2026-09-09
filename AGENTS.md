# Latest priority

- Paper writing (9 September): user requested a detailed handoff for another
  Codex. Start with docs/PAPER_WRITING_HANDOFF.md and newest HANDOFF.md. Keep
  historical force-PPO results, current direct-PVA simulation evidence and
  proposed battery calibration separate. Writing does not authorize restarting
  stopped fitting/PPO/MPPI or running flights. Include the documentation in the
  user-requested commit/push snapshot.

- Latest verified MPPI result (9 September): run20260909-135429-797997 is
  COMPLETED, with forward pull then backward release measured at interpolated
  contact: drone -0.62040m/s, tip +5.02739m/s, backward travel0.100000267m,
  hit1.26666754s/minimum distance1.34697cm. Current rolling horizon2s/60 actions,
  1024 samples, pullback initialization; separate5s maneuver limit, no optimizer
  iteration/wall-clock cap. First30 commands reused explicitly from provisional
  parent20260909-133459-410882, last9 refined (36 new iterations/56.934s).
  Full10.4333s recovery/hold and exact portable CSV/all-array replay pass;
  73 tests and native UI pass. No live worker or automatic rerun. Historical
  normalized M1 simulation only; fitting/PPO remain stopped, heartbeats PAUSED.
  Parent's interval-end contact check is superseded; old1.2s forward-only hit
  is not this task. Read docs/MPPI_PULLBACK_20260909.md and final audit
  runs/audits/mppi-pullback-contact-20260909. Older entries below are history.

- Latest user correction (9 September): a fast tip contact alone is NOT the
  requested strike. Drone must move forward to load the cable, then backward
  to release/propagate the cable toward the target. Previous1.2s result still
  had modeled forward drone speed~0.99m/s at contact; retain it under its old
  tip-hit semantics, not as verification of this newly clarified task.
  New MPPI task requires forward travel>=0.25m at>=1m/s followed by backward
  travel>=0.10m from the forward peak and drone speed<=-0.5m/s at contact,
  all projected onto strike direction using modeled ACTUAL drone pose/velocity.
  Existing tip-first/contact/speed/angle and physical limits remain. Phase
  rewards are bounded once per phase. Fitting/PPO remain stopped. Cold2s/2.5s
  diagnostics missed. NEW task-shaped initialization followed by rolling2s
  MPPI completed as20260909-133459-410882 under a provisional interval-end
  contact check. Corrected final result is recorded above. Preserve both runs;
  do not restart. Read docs/MPPI_PULLBACK_20260909.md.

- Latest completed MPPI improvement (9 September): user allowed longer horizon
  and tuning. Active lookahead now1.2s/36 actions, noise0.05, terminal geometric
  guidance off, explicit strike-exit costs. Cold1.2s/2s comparison favored1.2s
  in this seed. New rolling run20260909-131121-677417 hit at1.12667s/2.351cm,
  34 committed actions,168 iterations/381.85s. Complete recovery/export passed.
  Derived final replay20260909-131858-327997 retains EXACT same plan/CSV with
  corrected height filtering in recovery search; optimizer NOT rerun. Final
  drone peak1.72496m, complete10.26667s command; portable replay exactly equal.
  69 tests/native UI pass; GUI reopened. No live optimizer or automatic rerun.
  Model, hit rules, limits and raw task rewards unchanged. Historical M1
  simulation only; fitting/PPO/heartbeat remain stopped. Read
  docs/MPPI_LONGER_HORIZON_20260909.md. This supersedes the0.4s default below.

- Horizon correction verified (9 September): current MPPI uses a rolling
  0.4s/12-action lookahead, one committed 30Hz action per replan, separate 5s
  maneuver safety limit and no optimizer iteration/wall-clock cap. New run
  `20260909-124046-219074` COMPLETED: 150 actions, 938 optimizer iterations,
  971.71s runtime, simulated MISS at 0.514653m, no modeled constraint failure.
  Complete 13.7s whip/recovery/hold export passed; portable CSV and all arrays
  regenerated exactly. 49 tests and native six-page UI/rehearsal render pass.
  Desktop reopened with corrected controls. No live optimizer remains; do not
  restart automatically. This fixes horizon/state/replay semantics, not strike
  performance or PPO equivalence. Read docs/MPPI_RECEDING_PROGRESS.md.
  Fitting/PPO and both heartbeats remain stopped/paused; historical M1 only.

- Latest user instruction (9 September): horizon conflation is a BUG. Fix
  MPPI so0.4s is rolling lookahead, not the entire maneuver; fix the complete
  planning/recovery/UI workflow. Active implementation uses mppi.mode=receding,
  horizon_s=.4 and separate task.duration_s=5s maneuver safety limit; unlimited
  optimization with per-window plateau/manual stopping. Preserve model physics,
  real hit criteria and old snapshots. Explicit MPPI terminal guidance and
  exploration tuning are part of this authorized correction; no fitting/PPO.
  Read docs/MPPI_RECEDING_PROGRESS.md and latest audit status before launching.

- Latest MPPI audit (9 September): read docs/MPPI_MISS_AUDIT_20260909.md.
  User clarification pending:0.4s whole whip versus rolling lookahead. Current
  implementation is whole-whip open-loop planning. Historical PPO hit0.8733s
  under old force semantics; no trained same-contract PVA PPO baseline exists.
  Small exploration and an additional zero-action prior also contribute to
  stalled search. Diagnostic variants did not hit; no active reward/model/
  sampling changes adopted. Preserve old runs and await horizon intent before
  changing that contract. Fitting/PPO remain stopped.

- Latest user confirmation (9 September): remove MPPI's iteration ceiling.
  Active config uses mppi.iterations=0 (no limit); new MPPI defaults and UI
  support this. Stop only on practical reward plateau, manual stop or failure;
  no wall-clock timeout. Keep0.4s/12-action trajectory horizon and all other
  task/sampling settings. No new production run requested/launched for this
  change; old frozen runs retain their original finite budgets and results.

- Latest user correction (9 September): MPPI horizon is now0.4s (12 actions
  at30Hz), with the PPO-matched start/target and1024 random samples plus mean.
  Run20260909-121121-501019 COMPLETED:25 iterations/26.57s, simulated MISS,
  closest tip1.23165m. Recovery/export complete, portable replay exact. Evidence:
  runs/audits/mppi-horizon04-20260909. All other MPPI settings retained.
  No automatic rerun, fitting/PPO restart, or physical-flight validation claim.

- Latest user request (9 September, PPO-matched launch): set MPPI initial and
  target coordinates to PPO's saved values, [-2,0,1.255] and [-1,0,1.1] m,
  and run a NEW MPPI optimization. Run20260909-120300-901531 COMPLETED at31
  iterations/162.43s, modeled valid hit at4.400cm. Its current curved recovery
  planner could not find a return inside the saved limits: NO rehearsal/CSV/ZIP
  exported. Do not call this an exportable plan or restart it automatically.
  Same historical normalized M1, two-second
  MPPI horizon, rewards/limits/sampling. PPO config and all prior runs unchanged.
  No fitting, PPO or heartbeat restart. Audit: runs/audits/mppi-ppo-launch-20260909.

- Latest user decision (9 September, MPPI continuation): keep all fitting and
  PPO stopped while the user investigates the physical drone/height problem.
  Continue direct-PVA MPPI implementation and end-to-end verification. Bounded
  simulation diagnostics may use frozen historical normalized M1, explicitly
  labeled as historical simulation evidence, not the unfinished fresh M0 or a
  verified model of the repaired drone. No fit, PPO or heartbeat restart.
  MPPI verification completed; read docs/MPPI_PVA_VERIFICATION_20260909.md.
  Final selectable run20260909-115040-750483 retains the unchanged successful
  simulated strike from20260909-114422-355245 with corrected PVA recovery.
  Portable replay exact,43 tests pass; no live worker. Do not call it physical
  flight validation or resume diagnostics automatically.

- Latest user request (9 September, performance review): STOP the active PVA
  fit and queued PPO campaign, then optimize fitting, training and planning for
  speed/GPU efficiency. STOP files issued to both existing jobs; heartbeat
  paused. Preserve saved weights/optimizer state. No production restart while
  benchmarking. Preserve numerical/task/data semantics and measure speed plus
  forward/gradient parity before adopting acceleration. This supersedes the
  overnight auto-completion instruction below.

- User authorized overnight completion (9 September 2026): replace ACTIVE PPO
  and MPPI action generation with 30 Hz direct PVA via bounded XYZ jerk, no
  virtual force rollout; fit a FRESH preliminary M0 from the five normalized
  cf7/adp0 flights, with no historical learned priors/residual weights; rebuild
  and verify the whole UI around that workflow. Preserve all historical flight
  artifacts and force checkpoints under their original semantics. This
  supersedes the paused MPPI/force-only restrictions below. Isaac/PhysX remains
  PAUSED. Read docs/PVA_IMPLEMENTATION_PROGRESS.md. New PPO training is authorized
  as part of completing the new workflow; never resume old force checkpoints.

- Latest user instruction: future model fitting should use practical convergence/plateau stopping, not fixed NN budgets, and essential validation only. Read `docs/FUTURE_ADAPTATION_FITTING.md` before any new fit. Current normalized M1 is already fitted; remaining validation intentionally stopped. New PPO adaptation is running; inspect newest HANDOFF/status and do not duplicate or refit its frozen model.

- MPPI work is PAUSED at user request. Independent force code has targeted tests but its export audit was stopped; do not resume or call it fully verified.
- User authorized the bounded paired M0/M1 PPO study `runs/policy_adaptation/20260909-003414-347814-M1-policy-adaptation`. M0 control `20260909-003414-738407-seed655` and M1 `20260909-003414-507606-seed655` are COMPLETED, including saved rehearsals/exports. M1 recovery command peaks at 2.92688 m (predicted drone 2.73821 m); this height issue remains to review before real flight. No whip/recovery was silently changed. Check live study status; never duplicate/restart/extend it automatically. Each gets exactly +20,480 attempts from the retained checkpoint, with fresh optimizer/convergence state and the same flown launch/task/rewards. New study runs supersede the historical one-selectable-policy restriction; preserve the original stopped flight policy and all original artifacts.
- Adaptation progress now defaults to **Real flight progress**. User explicitly rejected held-out fit scores as adapted-flight RMS: M1/adp1 stays missing until actual new flights, and all PPO results are simulation-only. Keep held-out metrics only in **Model-fit diagnostics**. Original CSV-matched ghost stays immutable. Read newest HANDOFF and docs/M1_POLICY_ADAPTATION_PROTOCOL.md.

# Project instructions for the next coding agent

- **Latest MPPI design supersedes spline instructions:** user explicitly wants the same 30 Hz XYZ force actions as PPO. Active `MPPIPage` launches `planning/mppi_force_run.py` / `tools/plan_mppi_force.py`, using `plan_batch`, `reference_packets`, `execute_research_batch` and the seed's actual PPO action/reward/termination contract with selected M1. No maneuver spline, CEM objective or PPO training. New schema `mppi_force_fullstate_30hz_v1`; old spline outputs are historical only. UI distinguishes current launch/model results from explicit saved-run replay; never relabel/translate old CSVs. Default launch [0,0,1.225], target [1.5,0,1.1], one-second horizon inherited from selected PPO, editable. Read `docs/MPPI_FORCE_PLANNER.md` and latest handoff. Audit at the new launch was a predicted miss, not a flight-ready plan.

- **Latest UI decision supersedes CEM/eighth-page preservation:** user explicitly removed CEM and wants MPPI only. UI now has seven pages: MPPI Planner is sixth, Adaptation Check seventh. Do not restore CEM navigation. Saved CEM artifacts and shared legacy backend code remain for provenance/compatibility; the active planner uses `MPPIPage` with optimizer fixed to `mppi`. Latest MPPI launch is [0,0,1.225] m, target [1.5,0,1.1] m in `config/mppi.json`. PPO settings and saved artifacts are unchanged. No new optimization was started for this launch change.

- **MPPI is the requested new optimizer:** eighth page `MPPI Planner` defaults to published `data/model_candidates/20260908-adp0-M1/model.json`, with independent `config/mppi.json` and `runs/mppi`. Preserve the first seven pages, original PPO/M0 and independent CEM assets/settings. User prioritizes whip prediction; recovery prediction error alone does not reject M1, but complete command feasibility still applies. Read `docs/MPPI_PLANNER.md` and newest handoff. MPPI is offline spline-coordinate importance sampling, not aircraft feedback MPC or a renamed CEM elite update. Keep both residuals/model provenance and exact saved-CSV ghosts. No PPO resume or real flight occurred.

- **Adaptation Check is now the seventh page.** Preserve the existing six pages. `experimental_data/adaptation_check.py` and `simulator/gui/adaptation_check_page.py` load paired policy-scoped flight batches and show measured geometry against the exact CSV-matched saved execution prediction. Never substitute the command path or regenerate the ghost with a different/current model. Keep measured-log timing alignment and missing-marker gaps; no fitting/training during replay. Read `docs/ADAPTATION_CHECK.md` and newest handoff.

- **Current adaptation policy supersedes all older selections:** keep only `runs/ppo/20260908-195207-486249-seed655/checkpoints/best_validation.pt` selectable (SHA256 `d10657f471deb8b22cafbee9008792e8378c8c764ca97039d6dbe7afcc260eea`). The user selected it explicitly by screenshot for data collection. Run is STOPPED; do not resume automatically. Matching rehearsal is `20260908-203914-039721`; package `policies/PPO-adaptation-20260908-195207.zip`. Other PPO runs/top-level checkpoints and obsolete rehearsals/exports are recoverably archived outside the repo at `particle_filter_cable_project_archive_20260908_adaptation_policy_195207`; consult manifest.json. Latest/terminal no longer exist in the active run. Preserve retained provenance, all experiment data and independent CEM results. Workspace asset paths were repointed to identical-hash retained-run assets. Read newest handoff before interpreting historical cleanup/live-status notes.

- CEM launch settings are now independent: `config/cem.json:launch_setup` owns tracked-origin start and target; `config/launch_setup.json` is for PPO rehearsal. Never make CEM seed refresh overwrite edited launch coordinates or save CEM edits into PPO/SAC settings. New CEM jobs freeze launch coordinates and corresponding attachment position. This supersedes historical shared-launch notes below.

- PPO now supports `training_backend.type=isaaclab_model`: one Isaac Lab process hosts the existing external CUDA model and PPO collector. Keep both planning and fitted-execution phases, exact rewards/masks/termination, and both residuals unchanged. It is not PhysX replacement or a generic DirectRLEnv/RSL-RL migration. New GUI launches disable the old asynchronous NPZ replay feed. Read `docs/ISAACLAB_MODEL_TRAINING.md` and the latest handoff; do not restart the retained completed PPO automatically.

- CEM tuning now uses **Task & rewards** and `config/cem.json`. Freeze the resolved reward, hit criteria and strike direction in every new CEM job; never reuse PPO reward weights for CEM or silently change a saved run. Defaults match the original CEM objective. Save/load settings affect subsequent launches only. See latest handoff and `docs/CEM_PLANNER_20260908.md`.

- Latest user screenshot setup: tracked-origin **[-2,0,1.255] m**, target **[-1,0,1.1] m**. Shared native launch defaults are `config/launch_setup.json`; new PPO workspace attachment positions must include the unchanged body-frame offset. CEM has its own Rehearsal & Export tab. Saved artifacts retain their actual coordinates; never relabel or translate older outputs to match new defaults. Fresh outputs and tests are in the latest handoff.

- Latest CEM addition (8 September): the user authorized a **sixth CEM Planner page**, preserving the previous five. CEM optimizes quintic position splines through the saved drone/cable models and both residuals; PPO is only a rehearsal seed. See `docs/CEM_PLANNER_20260908.md` and latest handoff. Verified result `runs/cem/20260908-195210-552685`; package `policies/CEM-lower-setup-30Hz.zip`. Do not overwrite completed CEM jobs, confuse nominal candidate hit percentages with policy validation, or retrain/restart PPO as part of CEM. All 273 retained PPO files were verified unchanged. This supersedes older five-page instructions.

Read `HANDOFF.md` first; it is the current research and deployment decision record.

- **Cleanup supersedes historical policy-preservation/live-status notes below:** only working run `20260908-192755-046598-seed655` remains in the project policy library. It COMPLETED the requested +20,480 attempts at 314,368; do not auto-extend. Best checkpoint `best_validation.pt` selected at 308,224 (95.703125% development success). Old policy runs/exports/rehearsals are in `C:/Users/wts28/Documents/PHD/particle_filter_cable_project_archive_20260908_policy_cleanup`; resolve old references through its manifest instead of recreating old directories. Current rehearsal is `20260908-193608-967519-current-policy`, export `policies/Current-PPO-30Hz.zip`. Preserve raw/calibration/adaptation data. Read `docs/POLICY_CLEANUP_20260908.md`.

- Latest authorized continuation is **`20260908-192755-046598-seed655`**, **M0 flight PPO - lower setup - start X -2 target X -1**, from the selected flight policy at 293,888 attempts. Initial budget +20,480 to 314,368; 2,048 CUDA environments; both optimizers preserved; stopping history reset. Tracked origin [-2, 0.01287427254333901, 1.255] m, target [-1,0,1.1] m. Preserve this parent's original one-second objective, not stopped-v3 semantics. User rejected translated exports: translation prototype is archived, CEM paused, future commands must be generated in the actual lower setup. Read `docs/LOWER_FLIGHT_SETUP_20260908.md`; check live status before actions. The older v3 run is STOPPED at 80,896 and must not be resumed.

- User explicitly confirmed the flight/adaptation policy is **`20260908-152029-040639-seed655/checkpoints/best_validation.pt`**, not the newer PPO training run. Preserve this distinction. Match adaptation recordings to the actual executed FullState CSV and its saved rehearsal/model; do not infer the flown recovery version from checkpoint identity alone. Verified policy hash is in the handoff.

- Latest native rehearsal recovery uses a moving curved turn/descent and slow approach through `deployment/curved_recovery.py`; read `docs/CURVED_RECOVERY_20260908.md`. It supersedes compact stop/return braking. Whip commands and training are unchanged; preserve legacy recovery and old saved artifacts. Updated comparison rehearsal is `20260908-143530-158718` (older selected policy, distinct from current v3 training).

- Latest run supersedes older live-status notes: **M0 - Dynamic strike v3 - stop on execution hit**, `20260908-181630-424783-seed656`, is RUNNING from **44,032 attempts** of `20260908-174509-301420-seed656` (now STOPPED). The user corrected termination: first valid fitted-execution hit or five seconds; time cost 10 points/s through that event. PPO credit/replay stop there; offline CSV ends its whip at the next 30 Hz boundary and appends recovery. Same M0/both residuals and 2,048 environments. Both optimizers preserved; **stopping history RESET at 44,032**, with 20,000-attempt patience/minimum. Preserve this new history only for further SAME-objective continuation. Do not duplicate workers or resume stopped ancestors. Read newest handoff and `docs/EXECUTION_HIT_TERMINATION_20260908.md`.

- Preserve the user's historical selected PPO, saved experiment configurations, original calibration and measurements. The user authorized a separate native 30 Hz force / 30 Hz FullState workspace using M0 and both residuals on 8 September. Keep its five pages and preserve original 20 Hz semantics for legacy checkpoints.
- The user explicitly authorized all historical preliminary and whip data, including formerly protected `fig8vertical_002`, for the new complete-model fit on 7 September 2026. Preserve historical study splits in their snapshots. New all-data fits are development baselines; future recordings provide prospective assessment. Exclude suspicious intervals through recorded masks, preserving raw data.
- The user requested continued PPO training until reward convergence on 8 September. Authorized continuation `20260908-142124-902520-seed655` is running from 167,936 attempts with both optimizer states and built-in reward plateau stopping. Check live status before any action; do not duplicate a worker or resume after a user stop. Heartbeat `check-ppo-reward-convergence` monitors the run. A still-improving run reaching its 2,000,000-attempt review ceiling may be continued under this request using a new source/config snapshot and preserved optimizer state; update handoff/heartbeat. Do not silently restart crashes or launch unrelated supervisors. All older PPO/SAC runs remain stopped. Checkpoint hashes, stopping rule and results are in the handoff.
- The user shortened plateau patience/minimum additional attempts to **20,000**. A single authorized live monitor applies `early_stopping_override.json` without restarting the frozen worker; read `reward_convergence_override.json` and `reward_stop_monitor.json` for effective stopping state. Do not mistake the frozen worker's old 102400 setting for the active rule. Do not duplicate the monitor. Future continuations inherit the override.
- Latest drone-fit scope: use only the three historical whip takes generated through virtual force-policy rollout and cmdFullState execution for nominal drone response and drone NN fitting. Preliminary takes remain eligible for cable physics/residual fitting. Preserve pooled drone fits as historical comparisons; do not silently reuse them as the current candidate.
- Offline planning/export is not flight authorization. No ROS flight sender is implemented or validated. Ask for actual vehicle/interface details rather than inventing frames, firmware parameters or topic names.
- Preserve raw logs, failed trials and immutable study snapshots. Do not change rewards, physics or report selected validation as independent paper evidence during deployment work.
- Do not reset the working tree or rewrite Git history as cleanup. Superseded artifacts were moved outside the repo; permanent deletion was blocked by approval review, so do not retry via another mechanism.
- Prefer targeted existing tests and record what hardware/OS was actually tested. Do not claim Ubuntu/4080/5080 validation from a Windows/5090 run.
