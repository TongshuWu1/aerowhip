# Complete Technical Report: Sequential PPO Whip Control

Date: 2026-08-31

## Executive conclusion

The PPO result is real and scientifically useful, but it is not yet the
one-query, 49-D, open-loop policy architecture originally sought by the
project.

The successful experiment learned a **sequential state-feedback controller**.
Every 0.1 s, it rebuilds the current UAV/cable/target/physics context, appends
the remaining episode fraction, queries the actor, and emits a new 6-D
acceleration/body-rate command. A 10 s episode therefore contains 100 policy
queries and 1,000 physics steps. It does not emit one 49-D variable-duration
maneuver and does not use the Milestone-6A ACTIVE -> smooth SETTLE -> HOLD
command program.

The strongest from-scratch run completed 1,001,472 episodes. Its final
10,240-episode stochastic training window achieved 74.14% task success. The
saved best-validation checkpoint achieved 10/10 deterministic successes on
ten fixed, mildly varied initial states. A post-run audit performed for this
report evaluated all 512 existing physically propagated state-bank rows at the
canonical target:

- saved best-validation checkpoint: 408/512 = 79.69% current-task success;
- terminal checkpoint: 469/512 = 91.60% current-task success.

Those state-only results are promising. They do not establish a general
state-and-target policy. Training and every reported evaluation used one
canonical target and nominal physics theta. There is no held-out target-only
or joint state+target PPO result.

There is also a success-contract difference. PPO task success does not impose
the historical 0.50 m UAV-displacement gate; displacement is a continuous
reward cost. On the 512-state audit, the terminal checkpoint passed the
current task in 469 cases but passed all historical CEM hard gates in only one
case. The discrepancy was caused by UAV displacement, not speed, command
acceleration, or numerical failure.

The correct conclusion is:

> PPO has demonstrated a strong, fast, deterministic, canonical-target
> feedback controller under nominal simulated physics. It has not yet
> demonstrated the intended one-query open-loop map `(x0,g,theta) -> U`,
> target generalization, joint state-target generalization, theta
> conditioning, or performance comparable to the 98% CEM result under the
> same hard-gate contract.

## 1. Direct answers to the architecture questions

| Question | Evidence-backed answer |
|---|---|
| Is the policy queried exactly once? | **No.** It is queried 100 times during each 10 s episode, once every 0.1 s. |
| Is execution open loop? | **No.** The 83-D context is rebuilt from the current UAV/cable state before every action. This is closed-loop state feedback. |
| Does it use the 83-D `x0,g,theta` context? | **Partly.** It uses the same 83 learning features, recomputed every control step, plus one remaining-step fraction for an 84-D observation. |
| Does it output the final normalized 49-D action? | **No.** It outputs 6 values per step: local acceleration xyz and roll/pitch/yaw body rates. |
| Does it use variable `T_maneuver`? | **No.** The episode is a fixed 10 s and the policy may strike at any physics step. |
| Does it use ACTIVE -> 0.30 s SETTLE -> HOLD? | **No.** It continuously controls for the full horizon. |
| Held-out target success? | **Not evaluated.** Training and validation use only `[1.0,0.0,1.4]` m and direction `[+1,0,0]`. |
| Held-out initial-state success? | **Promising but development-only.** Original mild validation was 10/10. The post-run 512-state audit was 79.69% for the selected checkpoint and 91.60% for the terminal checkpoint. |
| Joint held-out state+target success? | **Not evaluated.** |
| Deterministic or stochastic success? | Both were measured. Training used stochastic actions; validation used deterministic `tanh(mean)`. The state-bank results are deterministic. |
| Episodes required? | 1,001,472 complete episodes, 100,147,200 control transitions, and 1,001,472,000 logical physics steps. |
| Policy inference latency? | Actor only: 0.129 ms median and 0.338 ms p95. State-to-action path: 5.142 ms median and 9.040 ms p95. |
| Nominal theta only? | **Yes.** Seven theta features exist in the input, but are constant at nominal values. |
| Was CEM used? | **No CEM actions, demonstrations, imitation loss, or initialization.** Policy, value, and optimizer began randomly. A fixed earlier context normalizer was reused as preprocessing. |
| Same production model? | **Yes for dynamics.** Training uses the pinned production UAV, causal residual, and 12-node DDER. |
| Same exact CEM deployment/evaluator contract? | **No.** PPO uses a sequential action interface, no 49-D decoder, no smooth settle, and a different validation batching path. |
| Same hard gates as CEM? | **No for reported PPO success.** Position/speed/direction/tip-first/finite remain; UAV motion limits are diagnostics and reward costs. |

## 2. What method was actually trained

The trained controller is:

```text
at control step k:

current UAV state
+ current distributed cable state
+ fixed target/direction
+ nominal theta
+ remaining episode fraction
        -> deterministic/stochastic PPO actor
        -> local acceleration xyz
        -> roll/pitch/yaw body-rate command
        -> 10 production physics steps
        -> observe new state
        -> repeat
```
The tested form is therefore:

```text
(x_k, g, theta, remaining_k)
  -> pi_PPO
  -> u_k
  -> production dynamics for 0.1 s
  -> x_(k+1)
```

It is not:

```text
(x_0, g, theta)
  -> pi_PPO
  -> complete U[49]
  -> one open-loop execution
```

Sequential PPO receives intermediate rewards and can correct its future
command using the evolving cable state. One-shot SAC and diffusion had to
discover an entire phase-sensitive maneuver at once from terminal outcomes.

## 3. Frozen physical model

The run explicitly requires:

`MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`

The environment uses `build_production_simulator`, which verifies the active
manifest and constructs:

- selected FullState UAV dynamics;
- pinned causal translational residual and fixed normalization/history;
- remeasured 0.9525 m cable geometry;
- rigid 55 mm attachment offset;
- 12-node DDER cable;
- selected bending stiffness and damping;
- validated PCG32 damping backend;
- CUDA float32 dynamics;
- configured DDER substeps and projections.

No PPO run modified UAV gains, cable parameters, residual weights, geometry,
or numerical integration settings.

Timing is:

- physics step: `0.01 s`;
- control interval: `0.1 s`;
- physics steps per action: 10;
- episode horizon: `10.0 s`;
- decisions per episode: 100.

## 4. Observation and conditioning

### 4.1 Core 83-D context

| Block | Width | Meaning |
|---|---:|---|
| UAV local velocity | 3 | Current translational velocity |
| UAV local quaternion | 4 | Canonical-sign xyzw orientation |
| UAV local angular velocity | 3 | Current angular velocity |
| Cable positions | 30 | Root-relative local c1...c10 positions |
| Cable velocities | 30 | Local c1...c10 velocities |
| Target position | 3 | Root-relative local target |
| Desired direction | 3 | Local unit strike direction |
| Physics theta | 7 | `Kp,Kv,ka,KR,Komega,log(EI),log(Cb)` |
| **Total** | **83** | |

The context is root-centered, yaw-aligned, normalized by the fixed normalizer
from the earlier one-shot SAC artifact, and clipped to `[-10,10]`.

PPO appends:

```text
remaining_fraction = 1 - control_index / 100
```

so actor and value function receive 84 dimensions.

Current state varies during every rollout, so within-episode feedback is
learned. However:

- gradient collection always starts from the canonical settled state;
- target and desired direction are constant;
- theta is nominal and constant.

The checkpoint can learn state feedback but has no demonstrated target or
theta conditioning. Retaining constant fields in the input is not evidence of
conditional generalization.

## 5. Action and execution semantics

The actor returns a tanh-bounded 6-D vector.

The first three values define a yaw-local acceleration. Its normalized norm is
radially limited to 1, scaled to a maximum of 10 m/s^2, and rotated to world
coordinates. The last three values are roll, pitch, and yaw body rates, each
scaled to a maximum of 4 rad/s.

For each 0.01 s physics step in a control interval, the command is:

```text
p_cmd = current simulated UAV position
v_cmd = current simulated UAV velocity
a_cmd = selected acceleration
q_cmd = current yaw-command quaternion
omega_cmd = selected body-rate command
```

Position and velocity are re-anchored at every physics step. The interface is
therefore closer to acceleration and attitude-rate control than to precomputed
FullState trajectory tracking. It permits roll/pitch/yaw response, but it is
not the CEM codec or 49-D maneuver representation.

## 6. Episode and single-attempt semantics

The task has one attempt:

1. Every physics step checks target distance for c1...c10.
2. The first marker entering the 50 mm sphere is permanently recorded.
3. If a non-tip marker enters first, a later c10 entry cannot recover success.
4. If c10 enters first while distance, directed speed, and direction pass, the
   episode records task success.
5. Success does not terminate the episode; costs continue to 10 s.
6. There is no permitted-strike timestep window.
7. Numerical failure invalidates the row. Failed rows are replaced by finite
   reset state, while post-failure transitions are excluded from GAE.

Successful entries in the 512-state audit ranged from 2.22 s to 9.45 s for the
selected checkpoint and 2.20 s to 8.99 s for the terminal checkpoint.

## 7. Success definitions

### 7.1 PPO task success

PPO-reported success requires:

- c10 is the first marker entering;
- c10-target entry distance <= 0.050 m;
- directed c10 speed >= 4.0 m/s;
- direction error <= 30 degrees;
- the rollout remains finite.

There is no time-window gate and no hard UAV-motion gate.

### 7.2 Historical CEM scientific success

The legacy diagnostic additionally requires:

- maximum UAV displacement <= 0.50 m;
- maximum UAV speed <= 3.0 m/s;
- maximum command acceleration <= 20.0 m/s^2.

These are logged, but are not PPO task-failure gates. Any CEM comparison must
keep this difference explicit.

## 8. Final reward

For control interval `k`, the implemented reward is:

```text
r_k =
    25 * delta_best_normalized_progress
  + 40 * delta_best_strike_quality
  + 150 * first_task_success
  - 25 * first_non_tip_entry
  - 15 * delta_robust_max_displacement
  - 1.0 * robust_displacement_integral
  - 0.25 * robust_speed_integral
  - 0.5 * normalized_acceleration_effort
  - 0.25 * normalized_body_rate_effort
  - 0.05 * action_change_squared
  - 100 * numerical_failure
```

### 8.1 Progress

Let `d0` be initial distance and `d_best` the best distance achieved so far:

```text
delta_progress = max(d_best_previous - d_best_current, 0) / d0
```

Only new progress earns credit, so moving away and returning cannot repeatedly
earn the same reward.

### 8.2 Strike quality

```text
quality = proximity(distance)
        * speed_score(directed_speed)
        * direction_score(cosine_error)

proximity = exp(-0.5 * (distance / 0.15)^2)
speed_score = sigmoid((directed_speed - 4.0) / 1.0)
direction_score = normalized sigmoid(
    (cos(direction_error) - cos(30 deg)) / 0.15
)
```

Only an increase in best-so-far quality earns reward. The direction sigmoid
was an important repair: the earlier `(cosine+1)/2` map awarded about 65%
credit to a 73-degree sideways entry; the final map gives it about 3% while
retaining a smooth gradient near 30 degrees.

### 8.3 Displacement and effort

Maximum displacement is charged through the increment in:

```text
log(1 + (maximum_displacement / 0.50)^2)
```

with weight 15. An additional time integral of
`log(1 + (displacement/0.50)^2)` has weight 1. The speed integral uses
`log(1 + (speed/3.0)^2)` with weight 0.25.

Acceleration and body-rate costs use squared normalized effort integrated over
the 0.1 s interval. Action smoothness is mean squared change in normalized
6-D action. None of these costs clips simulated motion or changes physics.

## 9. PPO model and optimization

### 9.1 Actor

```text
84
 -> Linear 256 -> Tanh
 -> Linear 256 -> Tanh
 -> Linear 6
 -> Gaussian mean
 -> tanh-squashed normalized action
```

- Actor parameters: 89,100.
- Six state-independent log standard deviations.
- Initial `log_std = -0.5`.
- Runtime `log_std` clamp: `[-5,1]`.
- Sampled log probability includes the tanh Jacobian correction.
- Deterministic action is `tanh(mean)`.

### 9.2 Value function

```text
84
 -> Linear 256 -> Tanh
 -> Linear 256 -> Tanh
 -> Linear 1
```

- Value parameters: 87,809.
- Total actor plus value parameters: 176,909.

### 9.3 Initialization

- Run seed: 242.
- Actor, value function, and optimizer were freshly initialized.
- Actor output-layer orthogonal gain: 0.01.
- Value output-layer orthogonal gain: 1.0.
- No policy checkpoint was loaded.
- No CEM action, elite, covariance, reward label, or trajectory was used.
- No behavior cloning or demonstration buffer was used.

The reused normalizer is preprocessing, not a learned action policy. The run
is pure on-policy PPO with the qualification that input normalization
statistics were not recomputed from zero.

### 9.4 Update rule

- Algorithm: clipped PPO with GAE.
- Optimizer: Adam over actor and value parameters jointly.
- Learning rate: `3e-4`.
- Adam epsilon: `1e-5`.
- Discount `gamma`: 0.99 per 0.1 s step.
- GAE lambda: 0.95.
- Policy clip ratio: 0.2.
- Value clipping: 0.2 around the old value.
- Value coefficient: 0.5.
- Entropy coefficient: 0.01.
- Gradient-norm limit: 0.5.
- Target approximate KL: 0.02.
- Early update stop at approximate KL greater than 0.03.
- Maximum epochs per collection: 4.
- Minibatch: 8,192 transitions.
- Replay buffer: none.
- Target networks: none.

Advantages are normalized over all valid collection transitions. Numerical
failure masks exclude post-failure data. The entropy term is the underlying
Gaussian entropy, not a learned SAC temperature.

## 10. Collection and compute

Each collection is:

```text
2,048 parallel episodes
x 100 control steps
= 204,800 on-policy transitions
```

After collection, PPO performs up to four shuffled passes. KL stopping can end
an update early.

| Quantity | Value |
|---|---:|
| Requested episodes | 1,000,000 |
| Completed episodes | 1,001,472 |
| Batch-aligned overshoot | 1,472 |
| Collections | 489 |
| Control transitions | 100,147,200 |
| Logical physics steps | 1,001,472,000 |
| Aggregate simulated time | 10,014,720 s = 115.9 days |
| Wall time | 6,454.66 s = 1.79 h |
| Throughput | 155.15 complete episodes/s |
| Final gradient-update count | 34,146 |

The billion logical steps were vectorized on CUDA, not executed as a billion
serial simulator calls.

## 11. PPO development history

### 11.1 Simple endpoint curriculum

The first sequential PPO counted endpoint distance/speed/direction without the
single first-marker semantics. It was stopped at 266,240 episodes with 51.79%
cumulative endpoint success. This showed that PPO could discover useful cable
authority in the 6-D interface, but it was not complete whip success.

A validated version stopped at 161,792 episodes with 23.97% cumulative
endpoint success.

### 11.2 Hard complete-whip continuation

A short continuation loaded the endpoint PPO policy and attempted an older
complete scientific task. It stopped at 12,288 episodes with zero success. It
was not from scratch and is not the final result.

### 11.3 Time-window reward study

Earlier balanced/compact runs retained an artificial success timestep window.
They produced zero complete success despite long runs. The timestep condition
was removed because the desired physics does not require a strike inside an
arbitrary interval.

### 11.4 Warm-started balanced studies

Several short balanced runs loaded the endpoint curriculum checkpoint and
therefore achieved high apparent success quickly. They explain the earlier
roughly 61% success after only 10,000-50,000 episodes. They are not evidence
of from-zero learning and are excluded from the final PPO claim.

### 11.5 Random-init high-displacement-cost attempt

A from-scratch run with maximum/integrated displacement weights 24/2 reached
only 21 successes in 282,624 episodes and was stopped. The motion penalty was
too dominant for sparse task discovery.

### 11.6 Final random-init run

The final run used displacement weights 15/1 and gate-centered direction
shaping. It is the only completed from-scratch PPO run used for the main claim.

## 12. Final training curve

Artifact:

`data/policy_training/whip_ppo_directional_displacement15_v1/2026-08-31T104735.951178Z`

Final totals:

- task successes accumulated during learning: 100,051;
- cumulative nonstationary training success: 9.99%;
- endpoint-only successes: 108,316 = 10.82%;
- historical-hard-gate diagnostic successes: 13,689 = 1.37%;
- final 2,048-episode batch task success: 75.44%;
- final batch endpoint-only success: 84.38%;
- final batch historical-hard-gate success: 0%;
- final 10,240-episode rolling task success: 74.14%;
- maximum batch task success: 77.20% at episode 999,424.

The cumulative 9.99% averages every stochastic rollout from the random policy
through the trained policy. It is not the final policy success rate. The final
rolling rate is the relevant training diagnostic.

Rolling 10,240-episode success first crossed:

| Threshold | Episode |
|---|---:|
| 1% | 704,512 |
| 10% | 722,944 |
| 25% | 833,536 |
| 50% | 940,032 |
| 70% | 989,184 |

Learning was late. The policy spent most of the first 700,000 episodes learning
proximity and tip-first behavior before satisfying speed and direction.

## 13. Original deterministic validation

### 13.1 Validation set

Ten states were fixed before training:

- source: existing nominal validation state bank;
- seed: 742;
- states at or below the 25th state-distance percentile;
- state-distance threshold: 0.2080;
- target: canonical fixed target;
- theta: nominal;
- policy: deterministic `tanh(mean)`;
- actual cadence in immutable run config: every 51,200 episodes.

They were not used for PPO gradients, which always reset to the canonical
state. They are a small development panel, not a sealed test. They use mild
nominal-simulation variation and the normalizer predates this run.

### 13.2 Validation progression

- Episode 153,600: median minimum distance 42.9 mm and 7/10 tip-first, but
  0/10 task success.
- Episodes 256,000-665,600: often 10/10 tip-first, still 0/10 task success.
- Episode 716,800: 9/10 current-task and 8/10 historical-hard-gate success.
- Episodes 768,000-870,400: regression to 6/10, 3/10, then 4/10.
- Episode 921,600: 10/10 current-task success.
- Episode 972,800: 10/10 current-task success.

The history is non-monotonic. Later policies produced faster, more directed
hits but used greater UAV displacement.

### 13.3 Selected best-validation checkpoint

The run selected episode 921,600 by current task success and then compactness:

| Metric | Result |
|---|---:|
| Current-task success | 10/10 |
| Endpoint-only success | 10/10 |
| Historical hard-gate success | 0/10 |
| Tip-first | 10/10 |
| Median minimum tip distance | 26.54 mm |
| Median successful entry distance | 41.46 mm |
| Median successful directed speed | 6.27 m/s |
| Median successful direction error | 22.36 deg |
| Mean maximum UAV displacement | 0.632 m |
| Mean maximum UAV speed | 2.358 m/s |
| Mean maximum command acceleration | 10.0 m/s^2 |
| Median success time | 2.33 s |

Ten successes in ten trials give a 95% Wilson interval of about 72.2%-100%.
The panel cannot establish an 80%-90% population rate by itself.

### 13.4 Latest scheduled validation

At episode 972,800:

- task success: 10/10;
- median minimum distance: 17.70 mm;
- median directed speed: 6.54 m/s;
- median direction error: 22.85 degrees;
- mean maximum displacement: 0.666 m;
- numerical failures: 0.

Training continued to episode 1,001,472. The original run did not validate at
that exact terminal episode.

## 14. Post-run audits performed for this report

These used existing checkpoints and state-bank data only. They did not train
or select a new policy, run CEM, access protected data, or execute hardware.

### 14.1 Canonical deterministic replay

| Metric | Best checkpoint | Terminal checkpoint |
|---|---:|---:|
| Current-task success | PASS | PASS |
| Historical hard-gate success | FAIL | FAIL |
| First marker | c10 | c10 |
| Entry time | 2.30 s | 2.28 s |
| Entry distance | 39.99 mm | 36.75 mm |
| Directed speed | 6.00 m/s | 6.97 m/s |
| Direction error | 26.79 deg | 23.13 deg |
| Maximum displacement | 0.619 m | 0.645 m |
| Maximum UAV speed | 2.216 m/s | 2.301 m/s |
| Maximum command acceleration | 10.0 m/s^2 | 10.0 m/s^2 |
| Finite | YES | YES |

Both policies solve the canonical PPO task and violate the old displacement
gate.

### 14.2 All-512-state canonical-target audit

Every row in the existing validation state bank was evaluated. The bank was
generated with nominal production physics and modest smooth UAV excitation:

- 512 states;
- 0.8 s generation horizon;
- excitation acceleration up to 3.0 m/s^2;
- accepted state speed up to 1.5 m/s;
- accepted displacement up to 0.3 m;
- real physical data: none;
- canonical target and nominal theta;
- deterministic actor.

This is broad **development** state variation, not a sealed final test.

| Metric | Best checkpoint (921,600) | Terminal checkpoint (1,001,472) |
|---|---:|---:|
| Current-task success | 408/512 = 79.69% | 469/512 = 91.60% |
| 95% Wilson interval | 75.99%-82.95% | 88.88%-93.71% |
| Endpoint-only success | 441/512 = 86.13% | 510/512 = 99.61% |
| Tip-first count | 435/512 | 469/512 |
| Historical all-hard-gate success | 6/512 = 1.17% | 1/512 = 0.20% |
| Median minimum tip distance | 30.86 mm | 22.94 mm |
| Mean maximum UAV displacement | 0.633 m | 0.674 m |
| Mean maximum UAV speed | 2.348 m/s | 2.364 m/s |
| Mean maximum command acceleration | 10.0 m/s^2 | 10.0 m/s^2 |
| Numerical failures | 0 | 0 |
| Median successful entry time | 2.385 s | 2.550 s |
| Median successful entry distance | 36.68 mm | 35.75 mm |
| Median successful directed speed | 6.37 m/s | 6.73 m/s |
| Median successful direction error | 23.86 deg | 23.09 deg |

The terminal checkpoint is better across the full bank, despite scoring only
6/10 in a post-run repeat of the original mild panel. This disagreement shows
that ten-state checkpoint selection was underpowered.

### 14.3 Hard-gate breakdown

| Gate | Best checkpoint | Terminal checkpoint |
|---|---:|---:|
| Finite | 512/512 | 512/512 |
| Maximum speed <= 3 m/s | 512/512 | 512/512 |
| Maximum command acceleration <= 20 m/s^2 | 512/512 | 512/512 |
| Maximum displacement <= 0.50 m | 9/512 | 1/512 |
| Current task AND displacement | 6/512 | 1/512 |
| All historical gates | 6/512 | 1/512 |

Displacement is the sole broad hard-gate failure. Terminal-checkpoint maximum
displacement had median 0.673 m, p90 0.741 m, p95 0.769 m, and maximum
0.887 m. Best-checkpoint values were 0.633, 0.698, 0.719, and 0.840 m.

This is not numerical runaway. All rows are finite, all stay below 3 m/s UAV
speed, and acceleration remains within the 10 m/s^2 action range. It is a
task-versus-motion-preference trade-off created when displacement became a
continuous cost instead of a failure gate.

### 14.4 Numerical-contract qualification

The audits use the same production model builder, but validation fixes the
residual batch to the logical validation size (10, 64, or 512). It does not
cyclically pad every query to Milestone-6A's fixed shape 2048. These are
production-model PPO evaluations, not exact Milestone-6A action/evaluator
replays.

## 15. Stochastic versus deterministic performance

Training actions are sampled from a tanh-squashed Gaussian. Final rolling
74.14% success measures local stochastic robustness around the current policy
on the canonical training state.

Validation uses `tanh(mean)` with no action sampling. The 10-state and
512-state results are deterministic and do not rely on repeated samples.
Nevertheless, the deterministic actor is still queried 100 times and receives
new state information after each interval.

## 16. Policy-query latency

The saved best checkpoint was measured after CUDA warmup over 1,000 serialized
batch-one queries.

### 16.1 Actor only

| Metric | Wall time | CUDA event time |
|---|---:|---:|
| Mean | 0.173 ms | 0.140 ms |
| Median | 0.129 ms | 0.113 ms |
| p90 | 0.257 ms | 0.233 ms |
| p95 | 0.338 ms | 0.244 ms |
| Maximum | 0.948 ms | 0.928 ms |

### 16.2 Current simulator state to action

This includes context construction, normalization/clipping, remaining-fraction
append, and deterministic actor:

| Metric | Wall time |
|---|---:|
| Mean | 5.798 ms |
| Median | 5.142 ms |
| p90 | 8.079 ms |
| p95 | 9.040 ms |
| Maximum | 13.505 ms |

This is below the 100 ms control interval. It excludes sensing, cable-state
reconstruction, communications, command transport, and hardware latency. It
is per control step, not one query for a complete maneuver.

## 17. Target, state, joint, and theta generalization

### 17.1 Initial-state variation

Evidence exists:

- 10/10 selected mild-state validation;
- 408/512 deterministic state-bank success for the selected checkpoint;
- 469/512 deterministic state-bank success for the terminal checkpoint.

This is the strongest current PPO generalization evidence. It uses nominal,
physically propagated simulator states and the canonical target.

### 17.2 Target variation

No evidence exists. All gradient and validation data used:

```text
target = [1.0, 0.0, 1.4] m
direction = [+1, 0, 0]
```

The target fields exist in the observation, but constant inputs do not teach a
conditional map. A target-switch evaluation alone would not substitute for
target-conditioned training.

### 17.3 Joint state+target variation

Not trained and not evaluated.

### 17.4 Theta variation

The seven theta fields remain in the context, but every episode uses nominal
parameters. There is no learned physics conditioning and no evidence that a
theta update would produce an appropriate maneuver change.

## 18. Comparison with production CEM

| Property | Production CEM | Current PPO |
|---|---|---|
| Controller class | One optimized open-loop maneuver | Sequential state-feedback policy |
| Online work | Thousands of simulator rollouts | 100 neural queries, no online simulator search |
| Input | Initial state + target + nominal theta | Current state + target + nominal theta + remaining fraction every 0.1 s |
| Action | Normalized 49-D | 6-D at each of 100 steps |
| Variable duration | Yes, 0.45-1.80 s | No, fixed 10 s episode |
| Smooth settle/hold | Yes | No |
| Target-only benchmark | 64/64 first seed | Not evaluated |
| State-only benchmark | 64/64 first seed | 469/512 current-task development result at canonical target |
| Joint benchmark | 62/64 | Not evaluated |
| Edge benchmark | 61/64 first seed, 62/64 with restarts | Not evaluated |
| Success gates | Complete production hard gates | Relaxed motion-gate task; hard gates diagnostic |
| Overall reported result | 251/256 first seed = 98.05% | 469/512 = 91.60% current task, state-only development |
| Complete historical gates | 252/256 within <=3 seeds = 98.44% | 1/512 for terminal checkpoint |
| Median planning/query time | 34.61 s | 5.142 ms state-to-action, repeated 100 times |
| Execution | Open loop | Closed-loop state feedback |
| Model needed online | Yes, for optimization | No for actor inference, but live state is required |

The percentages are not interchangeable. CEM was tested across target, state,
joint, and edge contexts under stricter gates. PPO was tested across state
variation only, at one target, using feedback and a relaxed motion contract.

## 19. Comparison with SAC and diffusion

### 19.1 One-shot SAC

One-shot SAC matched the desired 83-D-to-49-D architecture more closely, but
did not find scientific success reliably. It had to discover a narrow,
coordinated action manifold from terminal outcomes. Broad exploration left the
feasible tracking region; local spectral support improved feasibility but
remained task-limited.

### 19.2 Sequential SAC

Sequential SAC used the same 84-D/6-D environment class as PPO, but training
deteriorated and was stopped. PPO's on-policy clipped updates and GAE were more
stable for the final shaped task.

### 19.3 Deterministic CEM imitation

Action regression achieved low coordinate error but 0% held-out physical
success. Phase sensitivity and multimodality made coordinate MSE misleading.

### 19.4 Conditional diffusion

Diffusion retained the desired one-shot interface. A DDIM scheduler bug was
fixed, but candidate support and state conditioning remained incomplete. No
diffusion checkpoint was promoted.

### 19.5 Why PPO succeeded where these branches did not

PPO changed the learning problem from terminal one-shot trajectory search to
a dense sequential feedback MDP. It has:

- intermediate credit every 0.1 s;
- state correction throughout the motion;
- a 6-D action instead of a coordinated 49-D action;
- 100 adaptation opportunities instead of one query.

This may be a better control architecture, but it is an architecture change
when compared with CEM, one-shot SAC, and diffusion.

## 20. Reproducibility and artifact integrity

The run saved:

- immutable `config.json`;
- source-hash manifest;
- `RUN_CONTRACT.md`;
- latest and best-validation checkpoints;
- training CSV;
- validation history and state manifest;
- final summary/status;
- training and validation plots.

The current repository matches recorded hashes for the core runner, PPO
implementation, sequential environment, simulator/task/model inputs, and
normalizer. The checked-in validation plotting/cadence file and checked-in
config changed after the run to support shorter rolling windows and more
frequent future validation. The artifact config is authoritative: this run
validated every 51,200 episodes, not every 2,048.

The protected `fig8vertical_002` take was not accessed. Hardware was not
executed.

## 21. What has been proved

1. Conventional clipped PPO can learn a physically meaningful whip from
   random policy/value initialization in the nominal production
   UAV-residual-DDER simulator.
2. A deterministic feedback policy transfers from one canonical training
   initial state to a broad bank of modest physically propagated initial
   states at the same target.
3. Neural latency is comfortably below a 10 Hz decision interval on the
   current workstation.
4. Fast entries are not numerical runaway: audited rows are finite, UAV speed
   stays below 3 m/s, and command acceleration stays near 10 m/s^2.
5. Continuous displacement cost did not enforce the old 0.50 m preference;
   the policy trades additional displacement for task reliability.

## 22. What has not been proved

The current PPO work does not show:

1. one query followed by open-loop execution;
2. normalized 49-D action compatibility;
3. variable duration and smooth settle compatibility;
4. learned target conditioning;
5. held-out target-only success;
6. held-out joint state+target success;
7. edge-domain success;
8. learned theta conditioning;
9. exact fixed-2048 deployment replay for the sequential path;
10. robust satisfaction of the old 0.50 m displacement gate;
11. real-time distributed cable-state observation at 10 Hz;
12. real-hardware transfer.

## 23. Architecture decision

The evidence is not sufficient to replace CEM with the originally proposed
one-query PPO map:

```text
(x0,g,theta) -> pi_PPO -> U[49] -> open loop
```

That system has not been trained or evaluated.

The evidence does justify treating sequential PPO as the strongest current
learned-controller candidate:

```text
(x_k,g,theta,remaining_k) -> pi_PPO -> u_k
```

Whether it becomes permanent is a real architectural choice:

- If a reliable distributed cable/UAV state is available at 10 Hz, closed-loop
  PPO may be preferable because it corrects state error directly.
- If the system must observe once and execute open loop, this checkpoint is a
  useful result and possible trajectory source, but not the final controller.

## 24. Required comparison before promotion

Before changing the architecture or declaring PPO permanent:

1. freeze PPO architecture, reward, and checkpoint-selection rules;
2. train with physically propagated initial-state variation, not only validate
   it;
3. train over the established Phase-1 target domain;
4. split by state ID into train, validation, and sealed test;
5. report target-only, state-only, joint, and edge groups separately;
6. report deterministic success, not only stochastic training success;
7. report current task success and historical motion-gate diagnostics;
8. define one authoritative fixed-batch numerical contract for PPO;
9. record a nominal PPO action sequence and replay it open loop from varied
   states as a feedback-dependence ablation;
10. retain nominal theta initially, then add theta variation in a later
    dedicated milestone.

The feedback ablation is particularly informative:

```text
closed-loop PPO from varied state
versus
recorded open-loop PPO command from the same initial observation
```

If open-loop performance collapses, success depends on ongoing state feedback
and cannot be represented honestly as one-query maneuver generation.

## 25. Final summary

```text
Model:
    MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

Algorithm:
    CONVENTIONAL CLIPPED PPO + GAE

Initialization:
    RANDOM POLICY, VALUE, AND OPTIMIZER

CEM initialization/demonstrations:
    NO

Controller class:
    SEQUENTIAL STATE FEEDBACK

Open-loop one-shot policy:
    NO

Policy queries per episode:
    100

Observation:
    83-D CURRENT PolicyContext
    + 1-D remaining fraction
    = 84-D

Action:
    6-D PER CONTROL STEP
    acceleration xyz + roll/pitch/yaw body rates

49-D variable-duration action:
    NOT USED

ACTIVE / smooth SETTLE / HOLD:
    NOT USED

Episode horizon:
    10.0 s

Episodes:
    1,001,472

Final stochastic rolling training success:
    74.14% over 10,240 episodes

Original selected mild-state validation:
    10/10 deterministic current-task success

All-512-state development audit, selected checkpoint:
    408/512 = 79.69%

All-512-state development audit, terminal checkpoint:
    469/512 = 91.60%

Terminal checkpoint under historical complete hard gates:
    1/512 = 0.20%

Dominant hard-gate failure:
    UAV DISPLACEMENT > 0.50 m

Held-out target variation:
    NOT EVALUATED

Joint state+target variation:
    NOT EVALUATED

Physics conditioning:
    NOMINAL THETA ONLY

Actor-only median latency:
    0.129 ms

Current-state-to-action median latency:
    5.142 ms

PPO result:
    STRONG CANONICAL-TARGET FEEDBACK-CONTROL RESULT
    NOT YET A GENERAL ONE-SHOT OPEN-LOOP POLICY

Protected test:
    NOT EVALUATED

Real hardware:
    NOT EXECUTED
```
