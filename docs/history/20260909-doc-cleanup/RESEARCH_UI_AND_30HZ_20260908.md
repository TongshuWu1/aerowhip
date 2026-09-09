> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Research workspace and native 30 Hz PPO

The user authorized rebuilding the UI and training a new policy on 8 September 2026, after choosing to retain both fitted residuals. This enables a separate native 30 Hz workspace. It does not retime the selected historical 20 Hz checkpoint or replace the original calibration files.

## Five pages

1. **Model** shows the accepted M0 bundle, masses, geometry, effective drone parameters and directly sourced cable-tip error comparisons. All-data and leave-one-take-out results are labelled development assessments.
2. **Recordings** separates inspecting a take from importing another round and saving processing/review revisions. Collection instructions and historical preliminary-fitting tools remain accessible separately.
3. **PPO** provides named new runs, continuation, stop, a recoverable policy library, and task/reward settings. Curves show hit rate, median closest-tip distance, task return and model/reference failure rates. The 3D replay comes from the saved validation trial. New-run settings use the current workspace; continuation keeps the source checkpoint's model.
4. **Diagnostics** compares measured and predicted cable/attachment trajectories and shows time-dependent and per-marker errors. A second view compares actual logged FullState commands with aligned OptiTrack recordings, using the saved CSV-maneuver mask when available. It does not infer a desired cable trajectory from a drone command.
5. **Rehearsal & Export** selects a native 30 Hz checkpoint, tracked-origin start and target. An isolated GPU process generates the frozen plan and full reference. Inspect predicted motion, virtual forces, PVA, tracking error and the exact command table; save CSV or a portable policy/model/source ZIP. Opening a saved rehearsal is supported. Changing its setup invalidates the old preview and export buttons. Historical policies route to their original rehearsal.

No SAC page, flight sender, automatic fitting, data retirement or automatic policy promotion was added. Recording preparation still associates one reference with a round; per-take association and a complete corrected-route adaptation importer remain subsequent work. Review annotations alone do not make recordings fitting-ready.

## Model and clocks

Current pointer: `config/research_workspace.json`, configurations: `config/research_30hz/`.

The desktop launcher uses this workspace. The existing `run_simulation.py --headless` constant-force diagnostic retains its legacy configuration; use `tools/rehearse_research.py` for the new complete-model rehearsal.

Accepted source: `data/bootstrap_model_runs/20260908-064040-669809/bundle/`.

- Measured drone mass 0.157 kg; cable assembly 0.018 kg. Cable EI = 1e-7 N m², internal damping Cb = 1e-4 N m² s, fixed external drag = zero. Both fitted NNs are enabled.
- 30 Hz virtual force actions, 30 Hz native FullState commands, 150 Hz outer simulation and eight cable substeps: 1200 Hz cable integration, matching the fitted internal rate.
- The original 20 ms follow-through is retained. The frozen cutoff is rounded **up** to a native packet boundary, repeating the last force through the added partial packet. No further actor query or real-execution feedback selects this tail. The horizon remains one second in the initial new configuration.
- The policy plans with a hanging cable, known initial drone state and known target. Start and target vary within 5 cm balls. Small cable/velocity/estimation perturbations remain on execution. Force-gain, force-lag and physics-parameter randomization are zero in this selected fitted route.
- Planning and scoring remain separate. Actor queries occur only during virtual planning. The resulting immutable reference is scored through the effective loaded drone, its NN, the rotated attachment, and cable physics/NN. No second cable reaction is added to the fitted loaded drone.

The ideal offline launch has zero velocity, level tracking orientation, zero frozen hover compensation and a hanging cable. This is an explicit scenario assumption, not a measured real hover calibration or verified firmware mounting frame.

FullState position means **unshifted OptiTrack tracked origin O**. The preserved task centre is attachment A = [0, 0, 1.5] m. With the measured level-hover offset, its corresponding origin is approximately [-0.006655, 0.012874, 1.555] m. Target remains [1, 0, 1.4] m. The UI labels these references separately.

## One reference for training and export

`simulator/research_reference.py` integrates piecewise-linear virtual attachment velocities by the trapezoidal rule. It samples reconstructed position, virtual velocity and interval acceleration on the native 30 Hz clock. This avoids the acceleration ripple caused by imposing inconsistent numerical position/velocity pairs on a Hermite interpolant. The reconstruction distance is recorded.

The virtual attachment displacement defines desired origin displacement using the **initial** fixed offset. Future attachment rotation belongs to the execution predictor, which computes A(t) = O(t) + R(t)r. Future measured pose is never used to construct the plan.

`simulator/research_pose.py` uses the fitted delayed PD/feedforward translation, independent-scale-v3 attitude mapping, second-order SO(3) response and bounded drone NN. It splits integration at delayed packet events. `simulator/research_physics.py` captures the same direct cable equations on CUDA, selecting cuSOLVER because Windows MAGMA allocation is incompatible with graph capture. The captured tridiagonal solve uses `solve_ex` without a host synchronization; invalid states are rejected by execution scoring. No iterative-solver or renderer fallback was substituted.

The provisional command envelope checks a_z + g >= 2 m/s², tilt <= 60 degrees, specific-force norm <= 3.2/0.175 m/s² and speed <= 5 m/s. These are software screening limits, not measured airframe or room limits. Failed references/domain predictions are scored as failures. A later failure invalidates an earlier hit and its success rewards. Recovery is excluded from PPO scoring; reward coefficients and success gates remain unchanged.

`deployment/research_rehearsal.py` uses the same planning/scoring/reference path. It preserves every whip packet, appends analytic continuous-PVA braking, slow return and final hold, and predicts the complete reference separately. The complete-prediction whip prefix is checked against training before export. Recovery uses the existing gentle reference planner: acceleration transitions, 2.5 m/s² horizontal and 1.5 m/s² vertical braking, at least ten seconds for return, and three seconds of final hold. Return duration increases to respect its 0.4 m/s speed and 0.3 m/s² acceleration shaping limits.

Recovery is outside the historical maneuver fit. If its model prediction fails, playback stops at the last valid predicted sample and reports that failure; it does not substitute the command trajectory as a prediction. A dynamically screened reference can still be saved for offline review. This does not establish flight readiness.

CSV columns retain the existing twelve-column PVA/yaw convention. CSV starts at the whip and includes recovery/final hold. Intended external sequence remains takeoff → ten-second hold → complete CSV → land. The separate **virtual_force_30hz.csv includes gravity** and is labelled simulator input, not a direct force-controller command. No colleague controller/logger files were changed.

## Training and preservation

Fresh run: `runs/ppo/20260908-080833-956336-seed655`, display name **M0 - 30 Hz - both residuals - seed 655**. Target budget 500,000 attempts, collection batch 1,024, CUDA, seed 655. It starts with a fresh actor and no retimed checkpoint or embedded old force prior. Both residuals and their nominal parameters are frozen during PPO.

Each new native run owns its model assets and a source snapshot. This run executes from its own 158-file snapshot; later UI edits cannot change the running simulator. New contract guards reject silent retiming and unsupported uncertainty settings. Those guards were tightened after launch; the running snapshot already has the valid 30/30 Hz, zero-uncertainty configuration and unchanged runtime equations.

The audit is `runs/audits/20260908-072317-827885-research-ui-30hz/`. Of 133 entries in the initial preservation list, only the authorized source routing change to `learning/fullstate_rollout.py` differs. The other 132 entries, including original measurements, configurations, calibration and the selected historical checkpoint, remain byte-identical. Historical study/source snapshots remain intact.

Verification includes the targeted test log, native VTK checks/screenshots, two complete rehearsal probes, a moved-package replay and fixed-force batch comparison. The learned-force batch-1/batch-8 check differed by at most 6.63e-12 m in cable coordinates, 4.51e-13 m in origin coordinates and 4.16e-11 in PVA packet components. Both complete rehearsal probes gave zero training/export origin-prefix difference. These are software consistency checks; the earlier empirical fit and numerical-sensitivity limitations still apply.

**85 distinct targeted checks passed:** 83 in the recorded final set, one additional native-clock-contract regression and one native Windows launcher/VTK check. A focused follow-up rechecked affected UI paths after tightening that guard. The portable ZIP, extracted to another directory, regenerated bit-identical command, force, origin and cable arrays. An earlier broad legacy UI invocation was interrupted during slow unrelated work; its incomplete log is retained and is not presented as a full-suite pass.

Hardware tested: Windows 11, NVIDIA RTX 4080, Python 3.12, PyTorch 2.11.0+cu128. No Ubuntu, other GPU or real flight validation is claimed. Training status and validation evolve; consult the run's status and saved evaluation records rather than treating an intermediate percentage as its final performance.
