# PPO and SAC comparison

The paper comparison uses PPO and SAC under the same frozen physical baseline,
task, reward, action limits, initial-state distribution and validation scenarios.
Both prepare a force sequence using an initial estimate and execute it once,
then return to PID recovery. Actual strike feedback does not change commands
or their cutoff. MPCC is outside the proposed paper comparison.

Use independently trained seeds, shared evaluation budgets and an explicit
checkpoint-selection rule. Show current-policy validation hit rate and
hit-and-recovery rate as the main learning curves. Mean evaluation task return
is secondary; optimizer losses are algorithm-specific diagnostics.

All previous exploratory results and checkpoints were cleared on 2026-09-05.
See [the complete paper protocol](../PAPER_EXPERIMENT_PROTOCOL.md) for metric
definitions, uncertainty, data retention and export requirements. This document
specifies the next experiment design; it does not claim new comparative results.
