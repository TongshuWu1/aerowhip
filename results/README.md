# Current results

This directory is the curated, in-repository record of the methods that still
matter at the current stage. Full historical run directories are kept in the
dated sibling archive, not in the working repository.

| Method | Current conclusion | Main result |
|---|---|---|
| [PPO](ppo/README.md) | Preferred learned controller | 91.60% deterministic success on the 512-state nominal audit; continued 10 Hz feedback is materially more robust than compiled open loop |
| [SAC](sac/README.md) | Negative baseline retained for comparison | 106 successes in 1,206,272 episodes (0.0088% cumulative success) before the run was stopped |
| [CEM](cem/README.md) | Strong offline planner/reference | 98.05% first-seed and 98.44% up-to-three-seed authoritative success over 256 contexts; median planning time 34.61 s |

Shared context normalization and physically propagated state banks are under
[`common/`](common/). They are referenced by the active PPO, SAC, compiler-audit,
and CEM configurations.

The selected current architecture is **10 Hz closed-loop PPO**. CEM remains the
high-quality offline benchmark. SAC remains only as a reproducible negative
result.
