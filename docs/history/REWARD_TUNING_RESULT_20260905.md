# Selected open-loop strike: development result

The frozen PPO residual policy, initialized with a simulation-searched force
sequence, achieved **427/512 valid impacts (83.40%)** on the reserved
simulation scenarios (seed 190652). It was not tuned using these
outcomes. This is one training seed and provisional model uncertainty, not a
real-flight or replicated paper result.

| Reserved evaluation metric | Result |
|---|---:|
| Valid impact + settled recovery | 83.40% |
| Recovery, including missed strikes | 100.00% |
| Successful impact speed, mean magnitude | 5.652 m/s |
| Successful hit time, mean | 0.793 s |
| Peak drone travel per trial, mean | 0.872 m |
| Peak drone travel, 95th percentile | 0.912 m |
| Largest drone travel observed | 0.931 m |
| Varied-state/model cases | 299/384 valid hits |
| Nominal cases | 128 trials; 100.00% hits |
| Numerical failures | 0.00% |

The binomial Wilson interval for this simulator/scenario distribution is
79.9–86.4% (95%). It does not measure
between-training-seed variability or hardware uncertainty.

## What changed

The speed reward now uses world tip velocity projected toward the target.
Backward tip motion cannot earn positive forward-speed credit merely because
the drone retreats faster. Approach shaping has a bounded total budget, and
maximum drone excursion is penalized on both hits and misses.

| Reward term | Selected value |
|---|---:|
| Closest-tip approach improvement | 60 |
| Best near-target directed-speed improvement | 60 |
| First valid impact | 200 |
| Maximum-displacement cost weight | 15 |
| Displacement integral weight | 0.5 |
| Successful terminal displacement weight | 20 |
| Time cost | 0.1/s |
| Return / release bonuses | 0 / 0 |

The target remains a 5 cm sphere, with at least 4 m/s **world target-directed**
tip speed, at most 45 degrees velocity-direction error, and tip-first contact.
Impact speed above is velocity magnitude; the target contact force is not
modeled or measured. The active fitted DDER model was unchanged.

## Execution and initialization

The planner reads the initial drone and full cable state, privately simulates
the actor, and freezes the force commands before plant execution. The maneuver
is capped at 1 s. A fixed 20 ms hold of the last strike force allows a single
follow-through, capped by that same limit; it adds no actor query or repeated
swing. The subsequent PID controller receives state feedback. Strike contact
is used only for evaluation, never to alter the frozen execution schedule.

Recovery is allowed 15 s, using unchanged spatial, speed and dwell thresholds.
A paired test showed that 10 s was too short for cable settling, while all
128 development cases settled within 15 s without increased drone excursion.

Reward-only pilots did not produce valid deterministic validation strikes.
The successful initialization used **8,192 nominal CEM search attempts**.
The seeded development branches then consumed **8,192 PPO attempts**, including
discarded branch updates. Separate failed reward pilots consumed 10,240 extra
PPO attempts from a preserved 40,960-attempt parent. These counts exclude
diagnostic/evaluation rollouts; they are not a sample-efficiency benchmark.

The final guarded continuation rejected its proposed update. The checkpoint's
6,144 lineage-attempt counter includes that attempted update; its retained
actor comes from the selected 5,120-attempt checkpoint. Evaluation reuse is
explicit in the journal. The independent reserved evaluation was then run anew.

## Files and use

- UI run: **PPO · selected fast single strike**. Restart the UI to load code
  changes, select this run, then use **Run latest policy → Execute**. No new
  training run is needed to try it. Initial live-physics compilation can still
  take about two minutes in a fresh UI process.
- Selected run: `C:\Users\wts28\Documents\PHD\Projects\PRISMS\Sim2Real2SimWhip\runs\ppo\20260905-221711-285892-seed651`
- Reserved evaluation: `C:\Users\wts28\Documents\PHD\Projects\PRISMS\Sim2Real2SimWhip\data\reward_tuning_20260905\reserved_final_evaluation`
- Checkpoint SHA-256: `a37cd313b7a044fb5305a076b05ae27d42d709e8382377def5d5528513a45d89`
- Previous defaults: `C:\Users\wts28\Documents\PHD\Projects\PRISMS\Sim2Real2SimWhip\data\reward_tuning_20260905\defaults_before_publication`
- Detailed development protocol: `docs/history/REWARD_TUNING_20260905.md`.

Raw evaluation records, per-trial outcomes, actual first-trial replay, source
archives, package manifests, and unsmoothed PNG/PDF figures are retained.
Reward accounting, early impact-speed recording, live/training follow-through
parity, CUDA parity, PPO core behavior, UI compatibility and stop handling were
checked with targeted passing tests. No physical-fitting data or protected
recording was changed.

Remaining misses in the paired diagnostic were most sensitive to force-gain
errors. Real force calibration and measured uncertainty are the next evidence
needed before claiming sim-to-real performance.
