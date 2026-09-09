# PPO with known nearby targets and varied starts

## Larger-batch continuation

The user subsequently requested faster training through more parallel environments. The initial run was cooperatively stopped at 7,168 saved attempts. Continuation: `runs/ppo/20260906-174733-192436-seed652`, displayed as **PPO · 5 cm variation · batch 4096 · seed 652**. Collection batch increases from 1,024 to 4,096; development validation remains 256 trials and runs once per collection batch. The cumulative budget stays 500,000. Both Adam states, actor/critic weights, gradient-update count, learning rate, 4,096-transition optimizer minibatches and four optimization epochs are preserved. The child runs verified copies of the parent's frozen training source. New run metadata includes the parent checkpoint hash and the settings change. Checkpoint RNG states are not saved by this runner, so the sampling stream restarts; no bitwise continuation is claimed.

A warmed 40-step DDER CUDA-graph benchmark on this Windows/RTX 4080 measured 15,546 transitions/s at batch 1,024; 24,172 at 2,048; 31,706 at 4,096; and 35,701 at 8,192. All runs remained finite. Batch 4,096 gives about 2.04× the measured physics throughput; doubling again gives about 13% extra. These are physics measurements, not complete training throughput. Results: `data/performance/ppo_batch_scaling_20260906/physics_scaling.json`. The parent had averaged approximately 5.4 complete training attempts/s including earlier validation overhead. Larger collection batches alter PPO update cadence and validation frequency per attempt; assess learning curves as well as wall time.

Selecting a saved run in the UI now fills its batch, attempt budget, seed and device from its saved configuration, rather than showing unrelated new-run defaults.

Continuation verification: the initial child checkpoint exactly matched the parent's actor, critic, both optimizer states and gradient-update count. Its first 4,096-attempt collection and update then completed in **222.31 s**, approximately **18.42 attempts/s excluding validation**, with 70,720 valid transitions and no numerical failures. The learned checkpoint was saved at **11,264 cumulative attempts**, and updated-policy validation began. The parent's first 1,024-attempt collection/update measured about 8.92 attempts/s excluding validation; this suggests roughly 2× batch throughput, not a measured full-run speedup. Later parent rates around 5 attempts/s included intervening validation and are not directly comparable. Metrics are in `data/performance/ppo_batch_scaling_20260906/first_continuation_batch.json`. The UI saved-settings regression passed.

User-authorized experiment, 6 September 2026. Run `runs/ppo/20260906-171954-683453-seed652`, displayed as **PPO · target + start within 5 cm · seed 652**.

## Experiment definition

- Target center: `[1, 0, 1.4]` m; Euclidean radius 0.05 m.
- Physical initial cable-attachment center: `[0, 0, 1.5]` m; Euclidean radius 0.05 m. This refers to the attachment, not the tracked rigid-body origin.
- Retain the original mixture: 25% nominal attempts; 75% varied attempts. Within varied attempts the two position offsets are independent, uniform by sphere volume, rather than independent per-axis ±5 cm cubes.
- Existing initial cable-shape/velocity and physical/actuator uncertainty settings remain. For the spherical-start configuration, position-estimation error is applied to the estimate, so the physical attachment remains inside the requested sphere. Legacy cube configurations keep their old sampling law and RNG draw order.
- The target is known exactly for the current simulated attempt, recorded, and held fixed. The actor already has the target relative to its initial attachment in its 79-dimensional observation. No architecture or observation-size change is needed.
- PPO generates commands against a private nominal simulation from the initial estimate. Independent execution and its scoring use the same sampled target. No actual plant-state or contact feedback selects actions during the strike. Keep the original +X strike direction, 20 Hz holds, fixed follow-through and precomputed cutoff.
- Reward and active calibration are unchanged. A newly initialized PPO actor/critic/optimizer uses the existing embedded searched force prior; this is not a continuation of the missing selected PPO checkpoint.

## Budget and records

Training seed 652; maximum 500,000 attempts; 1,024 parallel environments on CUDA. Development validation uses 256 attempts and private seed 90652, initially and every 1,024 training attempts. Final holdout evaluation stays disabled. These are development measurements for one seed, not independent paper evidence.

The source configuration is `config/experiments/ppo_varied_target_start_5cm_20260906/`. Run-local `launch_config/`, `model.json`, `task.json`, and `ppo.json` preserve the exact experiment. The run executes its `source_snapshot/run_ppo.py` using the project virtual environment. `source_snapshot_manifest.json` hashes the code/config files; `launch.json` records the command and CPU-thread environment. `training.stdout.log` and `training.stderr.log` retain console output.

Per-attempt NPZ records now include the actual target XYZ and physical initial attachment XYZ. Validation scenario files include every target; trial outcomes and the first-trial replay identify their actual target, including when it differs from the nominal center. All attempts, including failures, remain in the logs. The new run is selected through the PPO run dropdown; the page displays the two variation radii. SAC stays disabled. The UI stop button uses the existing cooperative stop mechanism and preserves the last saved update.

The historical selected checkpoint remains a separate decision. A later export must explicitly identify a checkpoint from this new run and include its saved configurations. For real deployment, the desired target and measured initial drone/cable state must be supplied before planning, and the state must be rechecked before execution. This experiment does not implement or validate the hardware sender.

## Checks performed before launch

- 25 targeted tests passed: CPU/CUDA sphere bounds and repeatability, target observations, per-target independent scoring, frozen commands, saved validation targets, deployment timing, CUDA graph/eager agreement and validation preview behavior.
- 4 additional tests passed: a complete short new PPO run with varied positions, existing PPO/SAC temporary training/logging checks, and externally started run discovery in the UI. Temporary SAC regression testing did not start a production SAC run.
- A full 1,024-attempt CUDA rollout and four-epoch PPO update passed on Windows 11 / RTX 4080 with 18,147 valid transitions and finite rewards. Wall time was 114.22 s for this one check; it is not a full-training runtime estimate. PyTorch peak reserved memory was 0.305 GiB; this excludes other CUDA/driver allocations. Device-wide memory observed during the check was approximately 7,974 MiB of 16,376 MiB, including other device use.
- After launch, the native PPO viewer regression passed with the new run present. Initial development validation (before any PPO update) recorded 99/256 valid hits and no numerical failures. This is the fresh policy's existing-prior baseline on the new scenarios, not a learned improvement or physical-flight result. Read the run's live records for subsequent progress.
- The first production batch completed 1,024 attempts, accepted its PPO update, saved `checkpoints/latest.pt` and the per-attempt records, and entered updated-policy validation. All recorded first-batch targets and physical starts satisfy the requested radius bounds. Training continues in the background; current progress is in `status.json`.
