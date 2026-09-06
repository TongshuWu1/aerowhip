# PPO physics launch acceleration

The pilot's original eager physics averaged roughly 1.7 training attempts/s
in its early status reports. Twelve substeps and float64 physics are retained.
The change captures the existing DDER transition in a CUDA graph, then copies
the next state/force inputs and replays that graph. It does not change the
equations, precision, substeps, solver iterations or action hold time.

Graph output tensors are cloned before returning so saved previous states and
validation recordings cannot be overwritten by the next graph replay. Capture
is lazy for nominal planning and separate for a randomized execution plant;
the plant's sampled stiffness/damping constants are retained. Model references
avoid a closure cycle that could keep graph memory alive between evaluations.
CPU and saved runs without the flag use the eager path.

## Isolated physics benchmark

The previous pilot was gracefully stopped before timing. Each row compares ten
fixed-state steps after warmup; graph creation takes about one second and is
excluded from step timing. All comparison position differences were zero.

| Parallel environments | Eager step | Graph step | Physics speedup |
|---|---:|---:|---:|
| 512 | 286.9 ms | 85.8 ms | 3.34× |
| 1,024 | 309.0 ms | 93.5 ms | 3.31× |
| 2,048 | 317.5 ms | 122.7 ms | 2.59× |
| 4,096 | 347.5 ms | 185.0 ms | 1.88× |

This is a physics microbenchmark, not an end-to-end training speed guarantee.
It excludes reward bookkeeping, actor updates, validation and capture. Larger
batches improve throughput but change PPO's update frequency at a fixed attempt
budget; the continuation therefore retains batch 512.

Reproduce with `python tools/benchmark_gpu_physics.py` while the GPU is otherwise
idle. Source results: `data/ppo_audit_20260905/gpu_graph_benchmark.json`.

## Pilot continuation

Parent: `runs/ppo/20260905-181856-007834-seed651`, stopped at 1,536 attempts.
Continuation: `runs/ppo/20260905-184613-983374-seed651`, target 20,480 total attempts.
The saved model, task, weights and optimizer states are retained; graph execution
is enabled in the new immutable launch snapshot. The existing resume mechanism
reseeds RNGs, so this is not a bit-for-bit continuation of the sampling stream.
Parent checkpoints and histories are preserved.

The UI identifies GPU graph replay on the selected run. The runtime flag also
propagates to future UI continuations, without replacing their saved physical
model or task. Paper plots should retain both linked run histories.

CUDA tests compare curved states, changing force commands, retained histories,
randomized plant constants, different per-trial cutoffs, recovery state and
reward against eager execution. Existing open-loop and UI tests are also run.

The resumed policy reproduced all saved initial-validation metrics exactly on
256 scenarios, and policy/value/optimizer tensors matched the parent checkpoint.
The accelerated run completed its first 512-attempt collection/update in 60.3 s,
reaching 2,048 cumulative attempts. This excludes validation and is not a full
steady-state throughput measurement. No nonfinite states were reported in that
batch; 12.7% of attempted plans crossed the configured position limit.

## RTX 5090 larger-batch continuation

At the user's request for more CUDA parallelism, the graph continuation was
stopped at 7,680 attempts and resumed as
`runs/ppo/20260905-191047-020147-seed651` with 8,192 environments per collection.
The new initial checkpoint exactly matches its parent's terminal checkpoint,
including optimizer state. Float64 physics, 12 substeps, model, rewards and
learning rate are unchanged. The final partial batch keeps the 20,480-attempt
budget exact. This continuation changes PPO collection/update frequency and
must not be reported as a fixed-protocol independent research replicate.

An isolated graph-only benchmark measured 23,942, 29,622 and 31,797 environment
physics steps/s at batches 4,096, 8,192 and 16,384 respectively. Every batch
matched its eager reference exactly in position. Doubling 8,192 buys only 7.3%
more physics throughput while further reducing update frequency. These figures
exclude policy updates, scoring, validation and graph capture; they are not
end-to-end training rates. Results and reproduction options are recorded in
`data/ppo_audit_20260905/rtx5090_large_batches.json` and
`tools/benchmark_gpu_physics.py --batches 4096 8192 16384 --graph-only`.

The root PPO default is now 8,192. The UI shows the saved run's parallel count;
restart the UI to load the changed code. The training worker is already running
independently and should not be restarted. Initial validation showed 96% GPU
utilization; CUDA utilization does not imply peak arithmetic throughput or full
VRAM use. Double precision remains deliberate for the validated cable solver.

Verification: 16 PPO/acquisition/CUDA tests and 11 research-workflow tests passed,
including an actual CPU training run with a nondivisible attempt budget.

## Sparse plant execution

CUDA execution/recovery now gathers only rows with a nonzero frozen cutoff into
the physics graph. Results scatter back into the original batch for scoring and
recording. Planning, refused-plan rewards, executed forces, recovery duration,
precision and PPO updates remain unchanged. The all-executing case uses the
original full graph; the no-execution case still skips plant integration.
Set `compact_deployment_physics` false to retain the reference path.

Four CUDA tests passed, covering mixed, all and no executing rows, randomized
physical parameters, rewards, recovery outcomes and failure flags. A short
8,192-row benchmark with three executing plans measured 9.54–9.67 s for the full
path and 3.32–3.35 s for compact execution (about 2.9x). This includes graph
capture and only 14 physics steps, with PPO concurrently sharing the GPU; it
does not establish a whole-training speedup. Reproduce using
`tools/benchmark_sparse_deployment.py`. Results are in
`data/ppo_audit_20260905/sparse_deployment_benchmark.json`.

The current PPO process was left running and still has the earlier code loaded.
Stop at a saved batch and resume to activate the optimization. It also applies
to newly launched SAC workers through the shared execution implementation.
