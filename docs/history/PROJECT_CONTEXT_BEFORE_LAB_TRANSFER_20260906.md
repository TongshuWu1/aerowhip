# Project decisions to retain

Controller file received September 6: colleague supplied Downloads/force_controller.py.
It is a ROS 2/Crazyswarm2 Mellinger feedforward-hover test, not a custom Lee
implementation or PPO executor. Review: docs/CONTROLLER_INTERFACE_REVIEW.md.
Under the inspected upstream firmware, convert total world force using
a_ff=F_policy/ctrlMel.mass-[0,0,g_firmware]. Actual firmware/frame/massThrust
remain unverified. Offline mocks found 0.82 s setpoint gaps during gain switching
and six gains left zero after Ctrl-C during feedforward. No real ROS/hardware
execution and no edits to the supplied script were performed. Its 0.40 m takeoff
does not accommodate our 0.9525 m hanging cable. Vehicle identity and firmware
parameters have been requested; do not infer them from upstream defaults.

SAC baseline decision, September 6: user says SAC itself is not needed; the goal
is baseline comparison. SAC was intentionally stopped at 32,768 attempts after
validation collapsed from 48.8% to 0%. Saved critics predict initial values around
5,094 while actual validation return is -63.5; initial XYZ actions saturate.
No production reward/model/trainer defaults were changed and SAC was not restarted.
`results/20260906-sac-audit-baseline/` contains the reproducible audit, PNG/PDF
figures and per-scenario CSV. The initial SAC deterministic mean head is exactly
zero, so its evaluation is the fixed CEM prior baseline. Same 256 validation
scenarios: prior 48.8%, selected PPO 84.4%; on the 192 perturbed cases, 31.8% versus
79.2%. These are development validation results, not an independent paper test.
Recommend fixed-prior/PPO and later direct DDER optimization and adaptation
ablations; do not present a single collapsed SAC pilot as a fair algorithm ranking.
The existing supervisor's NEEDS_ATTENTION label is expected after intentional SAC
stop; preserve its completed PPO evaluation and do not restart the queue blindly.

Latest cleanup: retired files now reside outside the repo in the sibling
`Sim2Real2SimWhip-retired-20260906` directory, whose `moved.json` maps original paths.
Removed obsolete training/method/simulation/preview UI modules, MPCC reader and
configuration, duplicate `run_simulator.py`, eleven one-off audit/search/report
scripts, eight obsolete tests and one obsolete test helper. The still-used process
liveness and source-snapshot functions were extracted into small current modules.
Old artifact archives, stale initialization checkpoints, obsolete pivot fit and
previous source-review ZIP were moved out. Preserve current PPO/SAC study, selected
PPO snapshot, usable fallback, raw measurements, applied fit and CEM prior evidence.
The archive and dist paths mentioned below describe the earlier cleanup stage;
their current location is recorded by the external move manifest.

Repository cleanup, September 6: maintained tests now live under `tests/physics`, `tests/calibration`,
`tests/training`, `tests/flight`, and `tests/ui`; test root paths and shared-fixture imports were updated.
Dated audit/result documents are in `docs/history`; current guides stay directly under `docs`.
`archive/20260906_repo_cleanup/manifest.json` maps 62 moved items, including retired MPCC launcher/UI
files, completed migration/seed-launch scripts, old scratch files, and organized tests/reports.
Fifteen generated cache folders were also moved there after automatic approval review rejected their
permanent deletion with "blocked by policy". Do not retry that deletion through another mechanism.
The archive and generated results are ignored by Git/default search and pytest discovery. Current
training processes, frozen study source, all recordings and active fit/checkpoint dependencies were
preserved. Verification: 174 cases collected, 29 targeted checks passed. `run_tests.py` now accepts
one or more subsystem names; no argument still runs the full maintained suite.

Between-flight adaptation implementation (6 September UTC): see `docs/FLIGHT_ADAPTATION_QUICKSTART.md`.
User confirms tomorrow's logs include OptiTrack, commanded forces, and controller/IMU telemetry.
User is unsure about the file format and may connect ROS live to the simulator; ROS version/messages
are not known. `experimental_data.flight_recorder.FlightRecorder` provides a transport-neutral callback
logging sink for either ROS version. It publishes no commands and does not guess clock or frame transforms.
`experimental_data/flight_trials.py` imports an explicit normalized CSV/JSON contract, with immutable hashes,
causal launch initialization and pre-contact exclusion. `experimental_data/flight_adaptation.py` provides
boundary/coupled replay and bounded differentiable drag-only fitting (EI/Cb fixed per prior audit).
`learning/strike_adaptation.py` refines a recorded force sequence locally without changing actor weights.
`tools/adapt_flight.py` and the Real-world Updates GUI expose these offline operations. They never apply
the shared baseline automatically or connect to hardware. Every force candidate is flight_ready=false:
real controller mapping/response, next-launch initialization, independent uncertainty acceptance,
preliminary-data forgetting checks and full-strike physical validation remain outstanding. The residual
NN is deferred until repeatable held-out error supports it. Do not describe this first implementation as
flight-validated adaptation or a complete reproduction of the research proposal.

The user intends to return to adaptation research. The detailed proposal is
[RESEARCH_PROPOSAL_ADAPTIVE_AERIAL_WHIP.md](RESEARCH_PROPOSAL_ADAPTIVE_AERIAL_WHIP.md),
with the source review in [ADAPTATION_RELATED_WORK_20260905.md](ADAPTATION_RELATED_WORK_20260905.md).
No real-flight adaptation result has yet been established.

The deployment contract is initial drone and cable state, privately generated
force sequence, one frozen open-loop strike, then hover recovery. Real cable
and hit feedback do not alter the strike. The Lee-controller interface is pending.

Latest timing decision: **20 Hz policy commands**, 100 Hz physics, twelve DDER
internal substeps, one-second strike horizon, 20 ms fixed follow-through.
The earlier 30 Hz choice was superseded at the user's request.

Current authorized task: start fresh PPO and SAC comparisons at **500,000
attempts each**, using identical physics/reward/initial force prior. Use the
largest tested efficient batch on the RTX 5090, retain raw attempt records and
export rolling-smoothed plots to a results folder. The comparison queue runs
PPO then SAC so each receives the full GPU. Its protocol and source snapshots
are saved under results. Do not describe this single-seed study as a replicated
paper result. Preserve collapses and failures in the plots.

Active comparison: `results/20260905-235530-029769-ppo-sac-500k`.
PPO run: `runs/ppo/20260905-235530-202591-seed651`.
Read the comparison's `protocol.json` for the SAC directory and frozen worker
commands, and `status.json` for queue state. The detached queue exports plots
and performs final evaluations automatically. Its launch PID is in `launch.json`.
Stopping a training-page worker does not cancel the whole queue; the study-level
`STOP_REQUESTED` file cancels remaining work as well.

Cleanup is authorized for unused experiment artifacts. Preserve original and
protected recordings, active processed data, active physical-fit dependencies,
the CEM starting sequence, research notes and selected usable PPO fallback.
Automatic approval review blocked permanent recursive deletion. Twenty-eight
unused folders were moved reversibly into the comparison's
`unused_artifact_archive`; `archive_mapping.json` records their original paths.
No original recordings were deleted or moved.

The user subsequently authorized automatic validation-reward plateau stopping for BOTH current comparison algorithms. A detached tools/early_stop_comparison.py supervisor monitors immutable validation records without restarting or changing frozen trainers. No minimum total attempt count; patience five distinct validations, significant improvement two reward points; 500,000 is now a maximum. Best actual reward checkpoint is checkpoints/best_reward.pt; latest and all raw data remain intact. Exact mid-run amendment: study/early_stopping_protocol.json; supervisor PID: early_stopping_launch.json; live status: early_stopping_status.json. Per-run STOP_REQUESTED allows SAC to start after PPO; study-level stop cancels supervisor too. The supervisor waits for the original queue to release GPU before reserved evaluation of both reward-selected checkpoints. Original queue NEEDS_ATTENTION may mean expected early stopping; inspect explicit supervisor status. Do not silently relabel this as a predeclared fixed-budget study or a proven global optimum.

User correction: remove the initial 200,000-episode minimum. Only sustained lack of validation-reward improvement controls early stopping (five checks, currently 163,840 attempts); retain the 500,000 maximum. Both algorithms use this rule.

User manually completed PPO at 229,376 attempts and started queued SAC. Current PPO manual snapshot is checkpoints/manual_000229376_20260906T021921_current.pt in the current comparison PPO run. Its validation is 84.375% success, return222.5993; best reward remains separately preserved at196,608 attempts (223.8388). The supervisor now honors manual_completion.json for the reserved final evaluation; do not skip the manually completed PPO as a cancelled experiment. SAC started automatically and retains its500,000 maximum and plateau rule.

Publication preparation produced `dist/source-review-20260906.zip` and its checksum.
The source-only bundle excludes measurements, checkpoints and Git history; its
portable configuration copies preserve numerical physics/rewards/actions while
reducing development batch/replay sizes. Active configurations and SAC were not
changed. See `docs/history/SOURCE_RELEASE_REVIEW_20260906.md` for verification and
`docs/PUBLICATION.md` for outstanding authorship/license/data-release decisions.
Do not push the development Git history assuming ignore rules removed old data.
