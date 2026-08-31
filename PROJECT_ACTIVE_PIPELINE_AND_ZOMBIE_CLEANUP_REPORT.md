# Active Pipeline and Zombie-Code Audit

Date: 2026-08-31

## Outcome

The repository has two clearly separated active scientific paths:

1. the frozen production model and variable-duration CEM planner, retained as
   the authoritative open-loop reference;
2. an experimental sequential PPO whip learner using the same production UAV,
   causal residual, and 12-node DDER physics.

SAC, Figure-8 SAC, deterministic imitation, diffusion/scorer, and structured
diffusion experiments are historical evidence, not active training paths.
Their generated artifacts remain untouched.

## Current end-to-end pipeline

```text
recorded physical takes
  -> deterministic preprocessing
  -> decomposed UAV / residual / cable identification
  -> MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

canonical or physically propagated initial state
  -> production 83-D root-centered/yaw-aligned context
  -> sequential PPO policy (84-D observation including remaining-step fraction)
  -> each 0.1-s control step emits:
       yaw-local acceleration xyz
       roll/pitch/yaw body rates
  -> command re-anchored to measured p/v every 0.01-s physics step
  -> production UAV + causal residual + 12-node DDER
  -> complete 1,000-physics-step / 10-s rollout
  -> single-first-entry task event and fixed randomized-state validation
```

The policy can command attitude rates as well as acceleration; it is not a
velocity-only controller. It is queried sequentially during the simulated
episode, not once as a 49-D open-loop trajectory.

## PPO task contract

Task success requires the first target-entering cable marker to be c10 and, at
that entry, all three endpoint conditions:

- tip-target distance <= 0.050 m;
- directed tip speed >= 4.0 m/s;
- impact direction error <= 30 degrees.

There is no impact-time gate. The first entry consumes the only attempt, while
its integer physics step is recorded for diagnosis. Wall-clock or GUI/headless
speed never enters the task definition. The episode always continues to step
1,000, including after a successful strike.

UAV displacement, UAV speed, command effort, body-rate effort, and action
change are continuous reward costs. The historical 0.50 m / 3.0 m/s /
20 m/s² values are retained as validation diagnostics, not PPO task-failure
gates. Numerical non-finiteness remains a validity failure.

## Reward shaping

The active reward is bounded best-so-far shaping:

```text
increase in best normalized tip progress
  + increase in best bounded strike quality
  + one task-success bonus
  - one non-tip-first penalty
  - robust continuous max-displacement cost
  - robust displacement and UAV-speed integrals
  - bounded acceleration/body-rate/action-change costs
  - numerical-failure penalty
```

Progress and strike-quality terms can only pay when the episode best improves;
they cannot be farmed by circling near the target. Motion costs use
`log(1 + normalized_value²)`, so they remain smooth and informative without
creating huge PPO value targets for extreme but finite rollouts.

The earlier reward study compared three fixed variants from the same
deterministic random-initialization seed protocol:

- `balanced`: balanced task and compactness shaping;
- `compact`: stronger continuous displacement/speed preference;
- `strike`: stronger strike-quality and tip-first preference.

The final selected run uses robust maximum-displacement/integrated-displacement
weights 15/1 and a smooth direction score centered on the 30-degree task limit.
This corrected a compact sideways-entry behavior that received too much reward
under the old linear direction factor.

Policy, value network, and optimizer all start from scratch. No prior policy
checkpoint, CEM action, imitation target, or demonstration is loaded. Earlier
curriculum studies are preserved as historical artifacts but are not inputs.

## Validation and artifacts

The completed selected run:

- trained for 1,001,472 batch-aligned episodes from random initialization;
- achieved 74.14% success over the final 10,240 training episodes;
- achieved 10/10 deterministic success at the latest saved validation;
- uses the same ten fixed, mildly varied, physically propagated initial states;
- plots training task success, endpoint-only success, legacy numerical-gate
  diagnostics, and validation success;
- saves the best validation checkpoint by task success and then displacement.

Future runs validate every 2,048-episode collection batch, save checkpoints
every 10,240 episodes, and plot a 10,240-episode rolling training success rate.

After all variants finish, the automated audit evaluates each selected
checkpoint on 64 fixed physically propagated states spanning the existing
state bank and writes one comparison plot and report.

## Active entry points

- `run_simulator.py`: GUI and live Training page.
- `run_simple_ppo.py`: one PPO configuration, checkpoint/resume, validation.
- `run_whip_ppo_reward_study.py`: unattended sequential reward comparison.
- `tools/audit_whip_ppo_reward_study.py`: final broad validation and audit.
- `run_milestone3c.py`: frozen-model identification workflow.
- `run_milestone6a.py`: production CEM benchmark/reference.
- `run_milestone7c.py`: saved-action robustness analysis.

## Zombie cleanup performed

- Removed the just-created hard numerical-gate PPO configuration after the
  scientific objective changed.
- Removed its unused hard-safety reward field/path.
- Retained the old numerical gates only as an explicitly named diagnostic.
- Updated the GUI from the obsolete hard-gate run root to the three reward-study
  roots.
- Replaced stale README/report claims that Figure-8 SAC or no-learning mode was
  the active experiment.
- Preserved every historical model, report archive, CEM result, policy artifact,
  and interrupted/stopped-run checkpoint.

The large pre-existing working-tree deletions and archives belong to the prior
repository cleanup and were not reverted or broadened in this audit.

## Boundaries

- New CEM solves: 0.
- Production model modified: no.
- Protected `fig8vertical_002`: not evaluated.
- Hardware: not executed.
- Current PPO results are simulation-only and are not a promoted flight policy.
