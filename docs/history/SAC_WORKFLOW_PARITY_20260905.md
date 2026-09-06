# SAC training workspace

New SAC runs explore XYZ forces and use the shared model, reward and
initial-state-only planning/execution configuration. Existing run snapshots
remain unchanged, including older XZ-only actors.

The default collection is 1,024 attempts. Validation runs after each collection
and publishes the same first-trial 3D replay and learning-curve records as PPO.
The live viewport shares Run latest policy and manual Execute: one attempt,
then PID recovery, including predicted misses. Training and validation still
retain their existing success-gated deployment rule.

SAC reports planning and execution/recovery steps about every two seconds,
replay insertion, gradient update progress, and validation. Completed attempts
remain separate from within-batch progress. A smaller final collection now
respects a nondivisible target exactly. Nonfinite, position-limit and speed-limit
failure rates are recorded individually.

Shared CUDA graph physics and float64 integration are inherited through the
shared configuration. Actor and critics remain float32. The UI displays SAC's
own action axes and parallel count. No long SAC job was started during this
change; the running PPO worker was left intact.

SAC continuation restores networks, optimizers and temperature, but refills an
empty replay buffer. It is not an exact continuation of the previous replay
distribution. PPO and SAC have different update rules and batch defaults, so
paper comparisons should report attempts, transitions, gradient updates and
wall-clock time as well as held-out task performance.

Validation: 18 SAC/research-workflow tests passed, including a real short CPU
training run with collections of three then two attempts, intermediate progress,
gradient updates, validation recordings and exact five-attempt termination.
