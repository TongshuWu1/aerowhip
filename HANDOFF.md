# Sim → Real → Sim aerial whipping: lab handoff

Updated 6 September 2026. This is the current decision record for the next researcher/Codex. Read it before changing physics, rewards, policy selection or the flight interface. Historical reports preserve earlier decisions and are not instructions to restart those experiments.

## Immediate purpose and next steps

The user chose **Windows development (RTX 4080) and a small offline planning package on the colleague's Ubuntu flight computer (RTX 5080)**. The current development machine is Windows/RTX 5090. Keep the five-page workflow intact. Transfer the full development bundle to Windows and the deployment bundle to Ubuntu; create fresh virtual environments on each. See [lab setup](docs/LAB_SETUP.md) and [deployment contract](deployment/README.md).

The next lab session should establish the real vehicle/firmware/controller interface and produce synchronized logs. There is no implemented or validated learned-policy hardware executor yet. The desktop Execute button is simulation only. The deployment package verifies the chosen policy and generates a finite offline plan; it does not arm, publish ROS commands or declare flight readiness. Do not restart PPO/SAC merely because the machine changed.

Concrete next work with the colleague:

1. Confirm vehicle mass, firmware revision, controller equations/parameters, world/body frames, tracking origin and marker mapping. Resolve the received controller's transition/abort defects.
2. Run the deployment example offline and measure planning latency. Connect causal tracking initialization; verify current-state drift after planning. A slow plan can become stale even while the drone hovers.
3. Implement and test the ROS bridge, timing, acknowledged controller transition, logging and recovery. The user must decide/authorize actual flight operations separately.
4. Start with controller/force-response identification and appropriately scoped tests; measure achieved motion before attributing error to cable physics. Import both successful and failed trials into the adaptation workflow.

## Research question and claims

The intended contribution is reusable physical-model improvement plus rapid task-specific maneuver correction for aerial cable whipping. Preliminary motion-capture data establish a baseline simulator. A learned policy plans a force maneuver from the initial drone/cable state. Real execution supplies tracking, commanded forces and controller/IMU logs. Between flights, update the simulator from predictive mismatch, then exploit differentiability to correct a force sequence without retraining the entire actor every time. A better physical simulator may help other tasks; this transfer benefit is a research hypothesis to evaluate, not an established result.

There are two distinct updates: (1) shared physical simulator parameters, and (2) the task's force plan/policy. The implemented fast adaptation changes a sequence locally and leaves actor weights fixed. Retraining/distilling the actor later is a separate experiment. The user approved a future small neural **motion residual** after physical fitting, not a network that freely outputs changing EI/Cb. No residual network is active in the selected baseline.

The plant is a translational drone point coupled to a DDER cable. It does not simulate full UAV attitude, motors, thrust calibration or controller delay. Do not explain a real tracking error as a DDER parameter error until those effects and initialization have been checked. Ten centerline markers do not identify material twist. Contact dynamics are not identified by fitting arbitrary post-impact data, and success/failure labels alone are insufficient for physical fitting.

See [research proposal](docs/RESEARCH_PROPOSAL_ADAPTIVE_AERIAL_WHIP.md), [identification design](docs/IDENTIFICATION_RESEARCH_DESIGN.md), [related work](docs/ADAPTATION_RELATED_WORK_20260905.md), and [experiment protocol](docs/PAPER_EXPERIMENT_PROTOCOL.md). These distinguish planned experiments from implemented functionality. No real-flight adaptation or conference-level replicated performance claim has been established.

## Current physical baseline and initial state

Authoritative active values are in `config/model.json`; selected runs retain immutable copies. Baseline version is `20260905-235235-311763-20hz`. Canonical model JSON SHA-256 is `e759615bacd8c075d0fa6318f881d1a870ba6d00a0bdbbaf174660fca704c1c2` (canonical JSON hash, not the raw file-byte hash).

| Quantity | Current value |
|---|---|
| Drone point mass | 0.159 kg |
| Bare cable mass | 0.007 kg |
| Each of ten moving markers | 0.000909091 kg |
| Cable including markers | 0.01609091 kg |
| Total modeled mass / hover force | 0.17509091 kg / 1.71705527255 N |
| Bending stiffness EI | 2.8399396688594252e-8 N m² |
| Internal damping Cb | 3.752308484268439e-5 N m² s |
| Effective cable drag | 0.3 /s |
| Gravity | [0, 0, -9.80665] m/s² |
| Policy / physics / internal integration | 20 Hz / 100 Hz / 12 DDER substeps per physics step |
| Maximum strike / fixed follow-through | 1 s / 0.02 s, bounded by the horizon |

Attachment-to-C1 then successive marker intervals are `[0.063, 0.087, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1025, 0.1, 0.1]` m, total **0.9525 m**. The first interval has an interpolated midpoint. There are 12 simulation nodes: attachment node 0, midpoint node 1, C1–C10 at nodes 2–11. The attachment is a free pivot, not an attitude-clamped tangent.

The user measured the drone-top/propguard plane to attachment height with a ruler. The tracked rigid-body origin is at that top marker plane; it is not the attachment or center of mass. Body-frame tracked-origin-to-attachment offset is `[0.0066549972854827175, -0.01287427254333901, -0.055]` m. Rotate this offset with the measured rigid-body orientation before adding it to the tracked position. Lateral values came from constrained fitting; do not confuse the 55 mm height with the 63 mm attachment-to-C1 cable interval.

Physical parameter initialization and per-flight **state initialization** are different problems. The latter requires all node positions and velocities, consistent timestamps, causal derivatives and constraint projection. `experimental_data/state_initialization.py` provides a past-only 11-frame quadratic endpoint estimate and projected state. The flight CSV importer currently uses an 11-frame causal linear velocity estimate; these are not identical estimators. An online adapter must explicitly select/validate its estimator rather than claim they are interchangeable. Never initialize from future frames or assume a moving cable has zero velocity.

## Preliminary fitting and protected data

Training takes: `fig8_001`, `fig8_002`, `fig8vertical_001`, `osc_001`, `osc_002`. Development validation: `fig8_003`, `osc_003`. **`fig8vertical_002` is protected test data: do not open, plot, evaluate, fit or tune against its contents.** It is carried opaquely in the private development transfer. Roles are frozen in `data/dataset_manifest.json`.

The current recommended workflow fits lateral attachment geometry and effective cable drag while holding measured height, segment lengths, EI and Cb fixed. Earlier audits found weak EI/Cb identifiability and confounding with geometry/initialization. More optimization iterations cannot establish physical identifiability. The active fit is `data/calibration_audits/20260905_constrained_fit`, with `recommended_model.json`, candidate review and continuous 2 s / 5 s predictive evaluations. Its recommendation is `geometry_drag_seed`; baseline provenance also passes through `data/baselines/20260905-181108-847814`.

Use complete-take splits, causal initialization, measured-attachment boundary replay, consistent marker weighting and held-out long-horizon prediction. The raw preliminary recordings are motion measurements. Derived whole-system momentum-balance forces are estimates, not measured thrust. Do not tune a flexible residual or material parameters to compensate for a bad attachment transform. Candidate fitting does not apply the model automatically. Inspect held-out errors and then explicitly apply through the UI.

Historical full EI/Cb/NN fitting audit scripts and documents may name retired artifacts. They are not the default current fitting procedure. Some historical modules retain helpers imported by current calibration; do not delete them just because their CLI entrypoint is old.

## Policy and open-loop execution

The policy observation has 79 values: 36 relative node-position coordinates, 36 velocities, target relative XYZ, desired direction XYZ and remaining-time fraction. Position scale is 1 m, velocity scale 5 m/s, observation clipping 10. Its normalized XYZ action maps to **total world-frame thrust force in newtons**, including hover compensation; the simulator applies gravity separately. The current provisional bound is 3.2 N with nonnegative vertical force. This is not a verified hardware thrust envelope.

Current nominal attachment is `[0,0,1.5]` m, target `[1,0,1.4]` m, strike direction +X. A private nominal simulation queries the actor and compiles a sequence from the supplied initial state. Real strike execution receives no subsequent cable/motion/hit state. The sequence runs once and returns to normal control at its **precomputed** cutoff, even when it predicts a miss. The compiler stops at its modeled terminal event and adds the fixed 20 ms follow-through. Numerical invalidity rejects a plan. Never wait for a real success event to launch or to stop, never repeat the tail, and never silently convert this to real-state feedback.

In simulation, position PID establishes hover and recovers for up to 15 s after the strike. Actual attitude/rate stabilization remains necessary during force execution. Current launch drift guards are maximum node-position change 2 mm and velocity change 0.01 m/s since planning. They have not been validated against laboratory noise/latency. The offline export is CPU/eager planning, not a guaranteed real-time planner.

Reward definitions live in `learning/point_force_env.py` and the run's `ppo.json`. Current components: best normalized target approach ×60; best near-target directed speed quality ×60; valid hit +200; travel penalty based on `15*log1p((displacement/0.35)^2)` and integral weight 0.5; successful terminal travel weight 20; non-tip first contact -25; numerical failure -100; recovery failure -10; elapsed-time cost 0.1/s. Release-at-hit, return reward and continuous angle reward are zero. Hit radius is 0.05 m, projected target-direction speed at least 4 m/s, angle at most 45 degrees, with first-contact conditions. Read exact implementation for clipping/accounting. Do not retune these during the portability handoff.

## Selected PPO and stopped SAC

Selected run: `runs/ppo/20260905-235530-202591-seed651`.

Selected checkpoint: `checkpoints/manual_000229376_20260906T021921_current.pt`.

Exact file SHA-256: **`3c470ec4ec7b7cd3bec4312a507b7bca275c8c05eb87401287cff5c37a2f932d`**.

The user deliberately stopped PPO at 229,376 attempts and selected that current snapshot. Its latest/terminal copies have the same hash. Development validation: 216/256 success (84.375%), return 222.5993469. A slightly higher-return checkpoint at 196,608 exists; do not substitute it automatically. The transfer exporter explicitly verifies the user-selected checkpoint rather than choosing by a filename sort.

SAC run `runs/sac/20260905-235530-213563-seed651` was deliberately stopped at 32,768 attempts after collapse. Do not restart it or its comparison supervisor. Initial deterministic SAC equals the searched fixed CEM prior; initial validation success 48.828125%, then 0% after the first large update batch. The critic predicted values around 5,094 while actual return was -63.5, with saturated XYZ actions. This is evidence of this pilot's failure, not a general SAC/PPO ranking.

On the identical 256 development scenarios, prior/PPO success was 48.8%/84.4%; nominal 64 scenarios both 100%; perturbed 192 scenarios 31.8%/79.2%. Successful impact speed was about 5.68 m/s for both. The present evidence supports improved reliability over the fixed prior, not a demonstrated increase in successful impact speed. Count the prior's 8,192 search attempts and distinguish single-seed development validation from untouched paper testing.

Preserve `results/20260905-235530-029769-ppo-sac-500k` and `results/20260906-sac-audit-baseline` (raw histories, per-scenario CSV, PNG/PDF plots, audit and source snapshots). The former includes a completed reserved PPO evaluation; do not use reserved results for further tuning. Its supervisor's NEEDS_ATTENTION is expected after intentional SAC stop. There are no intended active training jobs.

Paper plots should retain raw attempts and report the smoothing window; show training reward and success against **attempts**, held-out validation against attempts, wall-clock/sample cost, impact speed, drone displacement and failure categories. Do not confuse optimizer updates, batches and episodes. Future independent seeds and uncertainty intervals are required. Preserve failed runs and avoid cherry-picked smoothing or best-checkpoint test selection. A useful additional baseline is per-initial-state direct DDER force optimization; it is not yet a completed comparison.

## Implemented between-flight adaptation

See [flight adaptation guide](docs/FLIGHT_ADAPTATION_QUICKSTART.md). `FlightRecorder` is a transport-neutral log sink; ROS topics and actual vehicle sender are not implemented. Imports enforce explicit time/frame contracts, immutable source hashes, causal launch initialization and pre-contact exclusion. Projection corrections above 20 mm are rejected. Protected data are refused.

Boundary replay drives measured attachment motion to isolate cable behavior; coupled replay drives the recorded force to check the combined model. Default fitting updates bounded positive cable drag (0.5–2 times prior), holding EI/Cb fixed. It uses SciPy trust-region reflective least squares, log parameterization, robust 2 mm position scale, equal trial weighting, prior regularization 0.05 and autodifferentiated JVP Jacobians. Candidate acceptance requires held-out boundary and coupled improvement and never auto-applies a shared baseline.

`learning/strike_adaptation.py` optimizes four XYZ tanh correction knots, bounded ±0.15 N per component, with 12 Adam steps. It rescans strict nominal hit/PID-recovery behavior and preserves the original sequence if the candidate is not better. It uses the recorded launch state; next-launch state conditioning and uncertainty-ensemble acceptance remain unfinished. Actor weights are unchanged. Every exported candidate remains `flight_ready=false`.

Synthetic mismatch audit in `data/adaptation_preflight/mismatch_audit`: planted drag 0.55/s, starting 0.3/s, fitted 0.5516321254/s. This is noise-free, same-simulator, short pre-contact data, not physical flight evidence. Future safeguards include actuator mismatch checks, preliminary-data forgetting checks, independent held-out trials, uncertainty assessment and full-strike testing. Add a small residual only after repeatable held-out motion errors justify it.

## Code map and workflow

| Area | Entry points |
|---|---|
| Desktop | `run_simulation.py`, `simulator/gui/main_window.py`, `simulator/workflow.py` |
| Physics | `simulator/point_mass.py`, `simulator/cable/dder.py`, `cuda_fixed_pcg.py` |
| Training | `run_ppo.py`, `run_sac.py`, `learning/point_force_env.py`, PPO/SAC modules |
| Frozen strike | `simulator/strike_plan.py`, `strike_sequence.py`, `live_flight.py` |
| Preliminary data | `experimental_data/processing.py`, `state_initialization.py`, constrained fitting modules |
| Flight adaptation | `experimental_data/flight_trials.py`, `flight_adaptation.py`, `learning/strike_adaptation.py`, `tools/adapt_flight.py` |
| Deployment export | `tools/export_lab_transfer.py`, `deployment/planner.py` |
| Verification | `tools/lab_preflight.py`, `tests/{physics,calibration,training,flight,ui}` |

UI pages remain Data & Calibration, Task & Rewards, PPO, SAC, Real-world Updates. No MPCC. PPO and SAC have separate plots/replay. Checkpoints and launch configs are immutable experiment records; active root configs affect new launches. Background jobs now use the running interpreter (`sys.executable`) on both OSes. CUDA NVRTC discovery supports Windows wheel DLLs and Linux wheel shared libraries; architecture is detected at runtime. Physics precision/integration equations were not changed for portability.

New lab-run defaults use device auto, collection batch 1,024 and validation cadence 1,024. SAC replay capacity is 262,144 and updates per collection 256. The root PPO status metadata now identifies these as lab defaults. These reduce workstation-sized allocations for the 16 GB 4080/5080; actual full training memory still needs checking on the target. **Batch changes alter update cadence and are not exact reproduction of the frozen 5090 study.** Saved run configs and policy files remain unchanged. Resuming a saved run can restore its larger batch: inspect settings before resuming. Do not promise full training fit from a one-step memory check.

## Repository state and cleanup boundaries

Obsolete UI/MPCC/duplicate entrypoints, scratch scripts/tests and superseded artifacts were retired; current regression tests are intentionally retained. Original measurements, protected data, applied calibration dependencies, selected PPO, stopped SAC evidence and CEM prior remain. Historical source snapshots belong to experimental provenance even when they contain old code.

Retired material is outside this project in sibling `Sim2Real2SimWhip-retired-20260906` and `Sim2Real2SimWhip-retired-lab-20260906`; their move manifests record original paths. Permanent recursive deletion was rejected by automatic approval review with “blocked by policy”; do not retry it through another mechanism. Reversible retirement cleans the working repository but does not reclaim disk space. These retired folders are excluded from the transfer.

The private lab bundles exclude `.git`, `.venv`, IDE machine settings and caches. Full-bundle active-run pointers are made relative; immutable experiment JSON retains original provenance strings, including old absolute workstation paths. Active runtime resolution should use bundled local paths; do not rewrite archived evidence to make historical provenance look local. Some historical audit commands require separately retired inputs. The source-only publication builder is a different export and deliberately has no data/checkpoints. Authors, license and public data release remain decisions for the researchers; do not publish or push the current Git history blindly.

Use Python 3.12, install the documented CUDA-enabled torch wheel, then requirements. PyCharm must use the new local virtualenv and project-root working directory. Do not copy Windows environments to Ubuntu. Local checks on RTX 5090 cannot establish actual RTX 4080/5080 or Ubuntu execution; run target preflight and offline planning there. Review `docs/LAB_VALIDATION.json` for the checks actually performed during packaging.
