# Multi-drone cable visualization

## Current training environment

For actual in-process training, use **PPO → New run settings → Train inside Isaac Lab — live model environment (native 30 Hz)**. PPO and the existing calibrated CUDA environment now run in Isaac's process and update the scene directly during collection. See [Isaac Lab model training](ISAACLAB_MODEL_TRAINING.md). The asynchronous workflow below is retained for older recorded batches only; the new launcher does not start it.

## Historical asynchronous batch viewer

Restart the research UI. Under **PPO → Training**, leave **Open live Isaac scene for this training run** enabled, set the run name/settings, and click **Train new policy**. Isaac opens bound to that exact new run. **Multi-drone scene → Open live training scene** can reopen a run's viewer later. Native 30 Hz continuations can use the same option. Legacy policies keep their old workflow.

The viewer now receives the **actual stochastic training collection**, not another deterministic policy rollout. Each displayed batch includes the sampled actions/virtual forces, the scored drone/cable execution, per-environment targets, hit/failure flags and the exact episode returns used for PPO credit assignment. It shows the attempt interval, full-batch mean return, selected-drone return, success count and current optimizer/validation status. The target sphere follows the actual batch success radius, including curriculum changes. Existing validation plots remain separately labelled deterministic validation; they are not the displayed exploratory batch.

The calibrated CUDA simulator and PPO still run in the project Python; Isaac runs as their live visual front end in its own environment. No PhysX physics replacement has been made. A batch is published **after collection, before the PPO update** and is replayed while optimization/validation/the next collection proceeds. This is an asynchronous view of real training batches, not per-physics-step streaming. Closing the viewer does not stop training. Rendering and recording add overhead; uncheck the live option for headless training.

Each run freezes the observer and renderer sources with its existing model/task/config snapshot. `live_scene/latest.json` commits one of two cache slots atomically; generation IDs, run identity and data hashes prevent mixed batches. Only the latest two trajectory batches are retained in this display cache, while `history.jsonl` retains batch provenance/metrics. Existing training attempt/checkpoint artifacts are unchanged. A stopped run is explicitly labelled stopped while its final collection remains viewable. Runs launched from an already-open old UI do not acquire this feature retroactively.

Verification: 23 targeted tests passed on Windows / RTX 4080, including exact recorder-on/off equality for actions, observations, rewards, masks, values, action log probabilities, CUDA random state and updated policy weights. A bounded four-attempt/two-update native integration test exercised an actual Isaac viewer following both successive training batches. Their displayed rewards and success rates match the training log exactly. Audit: `runs/audits/20260908-live-isaac-training/verification.json`. No production training was started or stopped by that audit; its active-run registration was disabled. Static 1,024-instance rendering was tested separately as recorded below; no claim of a sustained full-size live-training throughput benchmark is made.

## Saved checkpoint presentation mode

This is a presentation of the existing calibrated simulator, rendered by Isaac Sim. It does not replace drone/cable physics, retrain a policy, send flight commands, or alter a running training process.

## Use from the research UI

Restart the research UI to load the new **PPO → Multi-drone scene** sub-tab. The five main pages remain unchanged.

1. Select a native 30 Hz checkpoint, a scenario seed and 64, 256 or 1,024 drones.
2. Click **Generate batch replay**. The project Python generates deterministic open-loop executions in batches of 64 using the selected run's immutable sources, model, residuals and task settings. Failed/missed attempts are retained. This can take several minutes for 1,024 scenarios.
3. Click **Open Isaac scene**. Isaac runs in its own Python environment. Use Play/Pause, the timeline, playback speed, Grid overview, and Close-up with the drone index. Close Isaac before generating another batch or recording video.
4. **Record 30 fps MP4** renders one half-speed whip loop, retains lossless PNG frames, then encodes `replay.mp4`. Video sampling is fixed in simulation time, independent of rendering speed. The command/physics rates are unchanged. Each recording has a new folder and a provenance JSON.

Use **Open saved batch** to reopen a presentation without re-running the model. Results are stored under `runs/presentations/`. Isaac's interpreter path is editable and saved in `config/multidrone_viewer.json`; this machine has `C:/Users/wts28/env_isaaclab/Scripts/python.exe` (Isaac Sim 5.1.0, Isaac Lab 2.3.2).

## Meaning of the scene

- The policy generates a frozen virtual-force plan; the existing simulator creates the 30 Hz PVA reference, predicts fitted tracked drone position/rotation, and advances the cable with its NN residual.
- Playback includes the **whip only**. Looping resets the display to the recorded initial state; the reset is not a simulated recovery maneuver.
- Every cell corresponds to a separately sampled scenario. The saved distribution retains its nominal fraction, so some nominal scenarios legitimately coincide. No trajectories are cloned just to fill the grid.
- The tracked origin is O; the attachment is `A = O + R * offset`. Grid translations are added equally to O, cable nodes and target only for display.
- The drone mesh is illustrative. Cable display width is enlarged to 12 mm for visibility; saved cable geometry/physics retains the measured dimensions. The target sphere uses the 5 cm hit radius of the current task.
- Blue cable: executing; green: recorded valid hit; orange: finished without a valid hit; red: invalid execution. Invalid poses freeze at their last internally consistent frame. Final failure invalidates any earlier provisional hit.
- The renderer contains no rigid-body drone/cable simulation. USD instancing is used for drones and targets; the curves use the saved cable nodes.
- Saved checkpoint presentations are **deterministic checkpoint replays**, not recordings of stochastic training episodes and not prospective validation. Videos should carry this description in their caption. The live training mode above uses actual exploratory collections and is labelled separately.

## Command-line use

From the project directory, using PowerShell:

```powershell
.\.venv\Scripts\python.exe tools/export_multidrone.py --checkpoint <run>/checkpoints/best_validation.pt --output runs/presentations/<new-name> --num-envs 1024 --batch-size 64
& C:\Users\wts28\env_isaaclab\Scripts\python.exe tools/view_multidrone_isaac.py --replay runs/presentations/<new-name>
```

Render a video without changing the model:

```powershell
& C:\Users\wts28\env_isaaclab\Scripts\python.exe tools/view_multidrone_isaac.py --replay runs/presentations/<name> --record-dir runs/presentations/<name>/frames --fps 30 --speed 0.5
.\.venv\Scripts\python.exe tools/encode_multidrone_video.py runs/presentations/<name>/frames
```

Isaac startup and frame-capture diagnostics go into the replay folder's `isaac.log` when launched from the UI. Launching Isaac and training together competes for the same GPU; viewer performance is not training throughput. There is no automatic training pause, restart or monitor.

## Verification

Coordinate tests check that the display grid preserves drone–cable attachment and target displacement, and that damaged or inconsistent replay files are rejected. UI tests check loading without launching computation and preserve the five-page native launcher. The native Isaac smoke mode checks actual USD position arrays against the translated saved arrays, verifies zero rigid-body actors, exercises overview/close-up cameras and saves screenshots plus timing in `isaac-smoke-N.json`.

Physics migration remains deferred. Keep the rendering contract independent so a future validated physics backend can supply the same pose/cable arrays.

Completed on 8 September 2026, Windows / RTX 4080 / Isaac Sim 5.1.0: seven distinct targeted checks passed. Native rendering exercised 64 and 1,024 instances, including 90 changing frames at 1,024; maximum saved-to-USD position error was 1.91 micrometres after float32 conversion, with zero rigid-body actors. These short scene-update timings are not a sustained rendering or training benchmark. The first 64 trajectories of the larger recording exactly match the separate 64-case generation. Original frozen source files were verified unchanged.

Ready-made batch: `runs/presentations/20260908-multidrone-1024/`, using checkpoint `f3e5dd0cd6c582d8a7045405e83cb7118ead7337bf6407c08d20a1b437c37f4c` at 278,528 attempts. It contains 1,024 scenarios (954 valid hits, 70 misses, zero invalid attempts), without successful-only selection. The seed is 20260908; this is a presentation sample, not independent research evidence. `video-final/replay.mp4` is the verified 1600×900, 30 fps, half-speed grid video (61 frames). `video-30fps/` preserves an earlier interactive-camera capture; use `video-final/` for the fixed-camera export. Recording now runs headlessly with a fixed overview camera, so mouse navigation cannot change an export. Interactive playback retains free camera control.
