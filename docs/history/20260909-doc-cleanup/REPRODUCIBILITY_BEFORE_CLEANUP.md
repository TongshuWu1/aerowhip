> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Reproducibility and data availability

The source bundle contains physics, learning, recording interfaces, tests, a numerical baseline and an embedded CEM action prior. It contains no raw recordings, processed trajectories, fitted-data reports, policy checkpoints or private Git history. Real experimental data and trained-policy results therefore cannot be reproduced from source alone.

The bundled baseline retains the numerical physical values from the development configuration. Machine-specific provenance paths are removed. The manifest records original and bundled configuration hashes and each transformation. The embedded CEM prior is part of initialization, not a pretrained PPO/SAC actor; its preceding search used 8,192 attempts and must be counted separately in comparisons.

The portable bundle uses `device=auto`, 64 environments, a smaller SAC replay buffer and proportionate updates per collection for development. Those defaults are **not** the workstation's 32,768-environment comparison protocol. The physical model, action vectors, reward weights, task and timing are preserved. Match saved experiment settings before claiming reproduction of a reported run.

## Evidence to retain for an experiment

- Exact source snapshot, dependency versions and random seeds.
- Model/task/algorithm settings, initialization prior and any preceding search budget.
- Complete training-attempt records, validation scenarios and checkpoint hashes.
- Validation-selection/early-stopping rules and any mid-run changes.
- Final evaluation cases, including failures and recovery outcomes.
- For physical flights: original tracking, sent commands, controller telemetry, clock/frame mappings and event evidence.

Split data by whole recordings, not adjacent windows. Protected tests stay excluded from development. An empty dataset manifest in the source bundle is deliberate; do not substitute synthetic data while labeling the outcome as measured flight performance.

The maintained test suite includes both synthetic checks and local-data checks. The documented smoke subset is self-contained. For a source-only checkout, local-data tests are skipped with explicit reasons by `tests/conftest.py`; the absence of data must not be reported as a passing physical validation.

## Research claims

The current simulation comparison uses one training seed and a searched prior. PPO was manually stopped before SAC's maximum budget. It is not a replicated, fixed-budget algorithm ranking. Real-flight deployment/adaptation and cross-task transfer remain unvalidated. Author, ownership, data-sharing and licensing decisions are pending; consult [publication preparation](../../PUBLICATION.md).
