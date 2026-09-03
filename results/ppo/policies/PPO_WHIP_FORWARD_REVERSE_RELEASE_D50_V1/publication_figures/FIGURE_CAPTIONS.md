# Publication figure captions

1. **PPO learning curve.** Task-whip training success (raw batches, the configured rolling episode window, and cumulative success) together with deterministic validation on fixed physically propagated initial states. Success requires a tip-first 50-mm target entry, at least 4 m/s directed tip speed, and at most 30 degrees direction error. UAV numerical limits are not binary success gates.

2. **UAV-excursion trade-off.** Deterministic state-bank success and mean maximum UAV excursion from the initial position during compactness continuation. The dashed reference is the policy used to initialize continuation when available. Lower excursion is preferable; this panel deliberately exposes regressions rather than implying compactness improved.

3. **Optimization diagnostics.** Mean episodic reward, policy entropy, PPO approximate KL divergence, and clipping fraction. This is supplementary training-health evidence, not a task-performance result.

4. **Strike distribution.** Median and observed range across successful deterministic validation rollouts for tip error, target-directed tip speed, direction error, and strike time. Red dashed lines denote task thresholds; strike time is diagnostic only.

**Evaluation scope.** These figures describe nominal-physics simulation at the canonical target. Validation uses the run's fixed bank of mildly perturbed, physically propagated initial states. It is not target-generalization, broad state-distribution, hardware, or protected-test evidence. A figure generated before the run completes is marked `INTERIM_DO_NOT_PUBLISH` in `figure_manifest.json`.
