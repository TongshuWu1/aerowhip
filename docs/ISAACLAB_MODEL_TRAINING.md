# PPO training hosted in Isaac Lab

The PPO launcher can now start **one Isaac Lab process containing the PPO optimizer, environment collection and our calibrated CUDA drone/cable model**. The scene updates directly from that collection. It does not read saved batch files or run a second policy rollout to fill the scene.

## Use

1. Restart `run_simulation.py` after this update.
2. Open **PPO → Training → New run settings**. Name the run and set its batch size and budget.
3. Leave **Train inside Isaac Lab — live model environment (native 30 Hz)** checked. Select CUDA. Start a new policy, or select a checkpoint and continue it into a new run.
4. Isaac opens the live batch. Use **Overview**, or choose a drone number and **Close-up**. **Pause training** pauses collection/updates while keeping camera interaction available. Press it again to resume. **Stop and save**, or the PPO page's Stop button, preserves the last completed update and discards unfinished work.

The existing PPO page still provides learning curves, validation, policy management and checkpoint selection for rehearsal/export. Unchecking the Isaac option uses the project trainer. Legacy 20 Hz checkpoints require that original trainer; they are not silently retimed.

This computer's Isaac Python is `C:/Users/wts28/env_isaaclab/Scripts/python.exe`. The path can be changed in **Multi-drone scene → Isaac Python** and is saved in `config/multidrone_viewer.json`. Training snapshots include the selected backend, executable and source files. The child process removes inherited Qt plugin paths to isolate Isaac from the PySide GUI.

## Environment and clocks

`IsaacLabWhipEnvironment` extends the existing `PointForceWhipEnvironment`. `IsaacLabTrainingSession` hosts it using Isaac Lab's `AppLauncher`, `SimulationContext` and `InteractiveScene`. Our external CUDA solver supplies the states; the scene does not replace the fitted dynamics with PhysX. This is a custom external-model environment, not a conversion to Isaac Lab's generic `DirectRLEnv` reward/reset interface or its RSL-RL trainer.

The existing two-stage open-loop PPO contract remains:

1. **Virtual force planning:** the stochastic actor issues native 30 Hz actions. Its actual planning batch is displayed as each action advances the model.
2. **Fitted drone response:** the frozen FullState reference is passed through the nominal drone response and drone NN. The scene holds during this calculation; Isaac's event loop remains responsive between output steps.
3. **Fitted cable execution:** the predicted tracked pose determines the attachment, and the physical cable plus cable NN are stepped at 150 Hz, with the existing 1,200 Hz internal integration. By default, Isaac displays every fifth physics step. Episode events and returns are computed at the original physics resolution.
4. **PPO update and validation:** the collector assigns the execution return to the same included planning action, with the same masks and terminal rules. The scene holds while weights update and validation runs. The next collection resets the actual environments and uses the updated actor.

Both the currently configured fixed-horizon collector and saved native hit-or-timeout collector remain supported. No task, reward, action bounds, model parameter, residual weight, sampling distribution or termination setting is changed by enabling Isaac. Continuation inherits the selected checkpoint's configuration through the existing workflow.

The window labels the phases, collection number, hits and returns. Blue means an active displayed cable; green a provisional hit; red failure; orange a completed miss. The credited final return appears after collection finalization. Rewards are also recorded in the normal PPO logs. Validation remains the original evaluator; its batches are not substituted into the training scene.

Scene translations separate the environments visually and never enter the observations, reward, physics or export coordinates. Tracked origin and attachment follow the same `A = O + R * offset` convention. The virtual point-force stage uses an illustrative level drone; fitted execution uses the predicted attitude. Cable width is enlarged for visibility. Overview includes all environments, so use close-up to see an individual cable clearly.

Rendering runs as training advances, without forcing real-time playback. Initial CUDA graph capture can pause display updates. Rendering and GPU-to-CPU display copies add overhead; this feature is for observing actual training, not a claimed speed improvement.

## Evidence and files

- `isaac_runtime.json`: actual Python, Torch and GPU.
- `isaac_environment.json`: collection/frame/reset counts and current stage; `replay_source` is null.
- `isaac_collections.jsonl`: finalized batch reward, success count and PPO credit-rounding error.
- The standard checkpoints, attempt records, validation and training logs remain authoritative.
- `tools/check_isaac_training.py` runs a bounded baseline/hosted comparison and a real PPO update. It writes numerical verification and optional viewport captures under a chosen audit directory.

Tests on this machine used **Windows, RTX 4080, Isaac Sim 5.1.0 / Isaac Lab 2.3.2, Torch 2.7.0+cu128**. A visible 2,048-environment, full current one-second horizon comparison matched all rollout tensors, CUDA RNG, policy weights and value weights exactly against the ordinary collector in that same interpreter. It included two hits. A separate two-collection training run completed optimizer updates and wrote checkpoints at eight attempts. These are integration checks, not estimates of learned policy performance.

The short 2,048-environment comparison against project Torch 2.11.0+cu128 had identical actions/rewards and at most `1.49e-8` parameter difference after one update. This does not establish bitwise equivalence of long training across Torch versions. Audit folder: `runs/audits/20260908-isaac-training/`.

Twenty-three targeted tests passed, covering launcher/source isolation, GUI dispatch, terminal reward handling, cancellation and pose/cable frame consistency. An intentionally invalid startup configuration reported `FAILED` and process exit code 1; Kit fast shutdown does not hide the traceback. All 273 retained flight-policy files were verified unchanged. No long production training was started.

The older **Multi-drone scene** saved-presentation and recorded-batch viewer remains available separately. Its recordings are replays; enabling the new training backend does not launch that viewer or publish its NPZ batch feed.
