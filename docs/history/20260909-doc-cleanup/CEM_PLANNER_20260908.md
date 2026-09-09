> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Offline CEM spline planner

The sixth sidebar page, **CEM Planner**, optimizes a commanded position spline. It does not train or execute a force policy. The existing PPO pages, checkpoint bytes, physics and fitted residuals are preserved.

**CEM Planner → Optimize spline → CEM launch setup** edits tracked-origin start XYZ and target XYZ in metres. These are independent of PPO: CEM saves them in `config/cem.json` under `launch_setup`; `config/launch_setup.json` remains PPO's rehearsal default. Initial CEM values are **[-2,0,1.255] m** and **[-1,0,1.1] m**. **Save CEM launch setup** saves only these coordinates; **Save CEM settings** also includes them. Unsaved form values apply to the next optimization. Changing or refreshing the seed does not reset them. Existing replay files retain their own original coordinates.

Each new CEM job freezes its resolved launch coordinates in `cem.json` and derives its attachment start in `task.json` using the saved body offset. **Load settings from selected CEM run** restores that run's launch setup as well as its objective/search controls; older runs recover launch coordinates from their rehearsal metadata. PPO training, PPO rehearsal defaults, SAC configuration and saved checkpoints are not edited by any CEM launch control.

Use **Optimize spline** to choose a completed native 30 Hz rehearsal as an initial guess, name the run, review the tracked-origin start and target, and click **Optimize with CEM**. The seed's saved model and both residuals are copied into the new run. A previous CEM result can also seed another run. The seed is fitted and simulated again; its original success is not assumed. No trajectory is translated after generation.

The default search uses 64 parallel candidates, 12 iterations, 12 quintic B-spline control points, 2.5 cm control-point exploration, and 0.08 s duration exploration. Duration lies between 0.8 and 2 s by default; the UI allows a maximum of 5 s. Durations are quantized to 30 Hz boundaries. The first three control points coincide at the actual desired tracked-origin position, imposing zero initial velocity and acceleration. P/V/A are analytic derivatives of the same spline. Yaw and yaw-rate feedforward are zero. Target position is frozen before execution.

## Model and score

**CEM Planner → Task & rewards** exposes the hit bonus, distance cost, directed-speed bonus, time cost, jerk cost, invalid-contact cost and excess-height cost. It also exposes preferred height, distance/jerk clipping caps, hit radius, required directed speed, angle gate, first-contact and marker-order flags, and world strike direction. Tooltips explain each contribution. Nonnegative weights are required; zero disables a term. The hit definition and feasibility filters are independent of the reward weights.

**Save CEM settings** writes `config/cem.json`, including search controls on Optimize spline (population, iterations, elite fraction, spline points, exploration, duration, height and device). Current form values are used at the next launch even before saving; saving also preserves them across app restarts. **Load settings from selected CEM run** restores that run's frozen settings for a new run. Selecting or refreshing a seed does not silently replace edited criteria. Old runs without explicit reward fields resolve to the original default formula below. Each new run's `cem.json` contains the fully resolved reward and criteria, and `task.json` contains the matching hit criteria and normalized direction. PPO settings and saved artifacts remain unchanged.

For every candidate:

1. Evaluate analytic P/V/A on the native 30 Hz command clock.
2. Predict tracked-origin pose with the saved nominal drone response and drone NN, including its fitted command delay and attitude convention.
3. Compute the moving cable attachment from that predicted pose and the saved body-frame offset.
4. Run cable physics and the cable NN from the hanging initial state at 150 Hz.
5. Apply the new run's frozen strike gate. Defaults require a tip entry within 5 cm, directed speed at least 4 m/s, direction error at most 45°, and no earlier non-tip marker entry. User edits on Task & rewards override the seed's criteria in the new snapshot.

The PPO environment is reused only for initialization and contact/event bookkeeping. Its reward is not optimized and its force action does not drive this rollout. `planning/cem_execution.py` has a separate explicit objective:

`1000 * valid_hit - 100 * clip(closest_distance_m, 0, 10) + 10 * clip(directed_speed_at_closest_sample / required_speed, 0, 1) - 25 * termination_time_s - 0.001 * min(mean_squared_jerk_component, 10000) - 100 * invalid_first_contact - 20 * max(predicted_peak_height_m - 2.6, 0)`

Termination time is the predicted first valid hit time, otherwise the candidate's duration. The sampled jerk regularizer covers the proposed spline duration. Contact shaping is restricted to samples before termination. Command feasibility conservatively checks the full proposed spline, including any unused suffix. Earlier predicted hits are exported through the next 30 Hz boundary, followed by the existing continuous curved recovery and final hold. The real execution remains open loop: no real-time hit detector is introduced.

CEM updates a diagonal Gaussian from the best feasible elites, retains the incumbent, applies exploration floors, and saves up to 32 distinct leading candidates for final recovery checks. Population hit percentage describes optimizer samples of one nominal scenario; it is **not** a held-out policy success rate or experimental evidence.

## Feasibility and export

The saved model's speed, specific-force, positive vertical specific-force and tilt limits are enforced. These remain provisional simulation limits. Quintic polynomial extrema check the entire spline and recovery between command samples. Default height bounds are 0.08–2.8 m; predicted tracked origin and every cable node are also checked on the 150 Hz physics grid, including recovery and hold. There is no continuous collision/obstacle certificate or fitted-model uncertainty guarantee.

Only candidates passing complete reference and complete predicted drone/cable checks receive an export. An earlier whip hit cannot conceal a recovery failure. The complete-prediction whip prefix must reproduce the separately evaluated whip to within 1e-7 m. Recovery remains outside the empirically fitted maneuver scope.

**Rehearsal & Export** provides its own saved-run selector and **Start rehearsal** button, commanded trajectory, predicted drone/cable, PVA/tracking plots, the command table, exact CSV copy and a ZIP bundle. Entering the tab loads the selected completed CEM result; Start rehearsal plays that saved full prediction from the beginning. Optimization generates the rehearsal automatically. CEM has no virtual-force CSV and no pretend PPO checkpoint. The tracking-frame glyph represents the predicted drone origin/attitude; cable trails are at the attachment and tip. Color meanings are shared with the existing rehearsal.

Execution contract: take off → hold 10 s at the recorded tracked origin → execute the complete CSV once at its native timestamps → land. FullState positions refer to the unshifted OptiTrack tracking origin. Units are metres and seconds. The export is an offline artifact; no aircraft sender is implemented here.

## Saved state and process isolation

Each `runs/cem/<run>/` owns its settings, task, model assets, seed provenance, frozen source manifest, optimization history/candidates, selected spline, replay and CSV. Settings/coordinates are frozen at launch. PPO files are read only. Worker processes are isolated from Qt/VTK, as in the repaired rehearsal workflow. **Stop optimization** requests a cooperative stop; completed candidate history stays saved, and no incomplete CSV is offered. Closing the UI allows its already launched offline worker to finish. Opening the same prepared job twice is rejected by exclusive worker ownership; completed jobs cannot be overwritten by rerunning the CLI.

`planning/cem_run.prepare_job(...)` creates an immutable job and returns its frozen worker command. `tools/plan_cem.py --job <prepared-folder>` runs it. New planning should use a new folder, including when starting from an older CEM result.

## Verification

Tested locally on Windows / Python 3.12 / RTX 4080, not Isaac Sim or another computer. Targeted spline derivative/boundary, continuous envelope, CEM convergence/reproducibility, invalid-candidate, stopping, ZIP, existing recovery/export and native UI tests pass. Native Qt/VTK replay, rewind and PVA were exercised; review images are in `runs/audits/20260908-cem/`.

A broader legacy `test_research_workflow.py` check has a stale assertion that the target X equals +1; the user's current setup is −1. It is unrelated to the CEM optimizer and was not used to claim full-suite validation.

The final integration run `runs/cem/20260908-195210-552685` used 64 candidates × 6 iterations and finished in 111.54 s. It predicted a valid hit at 0.95333 s with 1.661 cm closest distance; complete CSV duration is 11 s. Maximum command/drone/cable heights are 2.458/2.276/2.589 m, and the separate-whip versus complete-prediction prefix difference is zero. The exact export is `policies/CEM-lower-setup-30Hz.zip`. All 273 existing retained PPO files match their pre-CEM hashes. These figures concern the single nominal simulation setup, not physical flight accuracy or robustness across initial conditions.
