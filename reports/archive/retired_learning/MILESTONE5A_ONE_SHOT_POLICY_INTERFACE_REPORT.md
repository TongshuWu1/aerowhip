# Milestone 5A — One-Shot Open-Loop Policy Interface Report

## Outcome

Milestone 5A passes. The repository now exposes a learning-facing interface with the strict semantics:

    one structured context
        -> one 49-dimensional complete maneuver
        -> one fully open-loop production rollout
        -> tuned task reward and compact episode metrics

No policy callback is accepted inside the physics loop. The future policy must produce the complete maneuver before rollout begins, so this API cannot accidentally become a sequential-feedback controller.

The unchanged simulator is `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`. No SAC components, training loop, replay buffer, CEM run, fitting, model adaptation, GUI change, protected-test access, or hardware execution were introduced.

## Implementation

The interface is intentionally small:

- `learning/policy_context.py`: structured invariant context and tensor schema;
- `learning/policy_action.py`: normalized 49D action encoding/decoding;
- `learning/one_shot_env.py`: batched complete-trajectory evaluation;
- `tests/test_milestone5a_one_shot_policy.py`: six focused interface gates.

Existing production code remains the single source for:

- FullState command integration;
- acceleration norm projection;
- variable-duration masking;
- tuned legacy `run_online` reward;
- feasibility and hard success gates;
- CUDA population rollout;
- deterministic trajectory replay.

## Policy semantics

The intended future call sequence is:

    context = build_policy_context(x0, goal, theta)
    normalized_action = policy(context.to_tensor())       # called once
    result = evaluate_open_loop_batch(context, normalized_action)

`evaluate_open_loop_batch` accepts a complete normalized action tensor, not a callable policy. It decodes the full trajectory before the first simulator step. No measurement, policy query, replanning action, or cable correction occurs during the rollout.

## Structured policy context

`PolicyContext` contains:

    uav_velocity_local_m_s                 [B, 3]
    uav_orientation_local_xyzw             [B, 4]
    uav_angular_velocity_local_rad_s        [B, 3]
    cable_positions_local_m                 [B, 10, 3]
    cable_velocities_local_m_s              [B, 10, 3]
    target_position_local_m                 [B, 3]
    target_direction_local                  [B, 3]
    physics                                 seven raw parameter batches

The ten cable observations are exactly `c1...c10`, corresponding to production DDER nodes `2...11`. Both cable position and cable velocity are mandatory.

The context also carries non-learning rollout provenance:

- the complete initial production simulator state, including the residual FIFO;
- the world-to-policy frame transform;
- the command position, velocity, and yaw at the policy-query boundary.

These provenance fields are required to decode and simulate the maneuver but are deliberately excluded from `context.to_tensor()`. Raw global UAV position is therefore not duplicated as a network feature.

## Context tensor schema

The context tensor dimension is **83**.

| Slice | Feature | Width | Encoding |
|---:|---|---:|---|
| `0:3` | UAV linear velocity | 3 | yaw-local m/s |
| `3:7` | UAV orientation | 4 | local active xyzw quaternion |
| `7:10` | UAV angular velocity | 3 | yaw-local rad/s |
| `10:40` | c1...c10 positions | 30 | flattened root-relative local coordinates |
| `40:70` | c1...c10 velocities | 30 | flattened yaw-local world velocities |
| `70:73` | target position | 3 | root-relative local coordinates |
| `73:76` | target direction | 3 | local unit vector |
| `76:83` | physics context | 7 | five linear gains, log(EI), log(Cb) |

The authoritative slice metadata is returned by `policy_context_tensor_metadata()`; downstream learning code does not need independent hard-coded indices.

## Local coordinate frame

At policy-query time:

    origin = physical connector/root position
    local +Z = world +Z
    local yaw = initial UAV world yaw

World vectors are rotated by the inverse initial-yaw rotation. Positions are first translated by the physical root position and then rotated. This preserves gravity alignment while providing translation and global-yaw invariance.

UAV orientation is represented as active body-to-local xyzw. Quaternion sign is canonicalized by making the largest-magnitude quaternion component nonnegative, so `q` and `-q` produce one representation.

Angular velocity uses the existing world-frame state rotated into the yaw-aligned local frame. Cable velocities are likewise rotated into the local axes; they are not removed from the context or replaced by finite-difference positions.

Numerically insignificant canonical-hover yaw drift at or below `32 * dtype epsilon` is treated as zero. Physical yaw outside this floating-point tolerance is never quantized.

The algebraic invariance test confirmed that equivalent synthetic global translations and yaw rotations produce matching 83D policy features.

## Physics context

Raw structured values are:

| Parameter | Frozen value | Learning feature |
|---|---:|---:|
| `K_p` | 4.020097778647703 | linear |
| `K_v` | 12.05728865003419 | linear |
| `k_a` | 0.7327301468333923 | linear |
| `K_R` | 69.18419375930429 | linear |
| `K_omega` | 11.456583174321011 | linear |
| `EI` | 9.225797579462985e-05 | `log(EI) = -9.290921820302584` |
| `Cb` | 0.0027704569956681496 | `log(Cb) = -5.888742992005043` |

The frozen UAV residual remains active in the simulator but its neural-network weights are not policy-context features. Milestone 5A rejects a context whose seven raw parameters differ from the active frozen simulator; physics randomization is not enabled yet.

## Policy action

The complete normalized policy action is:

    z_normalized in [-1, 1]^49

It contains:

    acceleration knots in policy-local axes    16 x 3 = 48
    maneuver duration T                         1
    total                                      49

The deterministic physical decoder performs:

    raw_acceleration_component = 20 * clamp(z_component, -1, 1)

and then applies the existing per-knot vector-norm projection:

    ||a_k|| <= 20 m/s^2

It does not interpret each axis as an independent physical ±20 m/s² limit.

Duration decoding is:

    T = 0.45 + 0.5 * (clamp(z_T, -1, 1) + 1) * (1.20 - 0.45)

Therefore:

    0.45 <= T <= 1.20 s

Local acceleration knots are rotated into world coordinates with the query-time yaw frame before FullState construction. A physical-action encoder is also provided for deterministic reference and future teacher-data ingestion.

## FullState command generation

The decoder reuses `planning.variable_duration.variable_duration_fullstate`. Acceleration knots are linearly interpolated in normalized maneuver time and consistently integrated to produce:

    a_cmd(t) -> v_cmd(t) -> p_cmd(t)

The policy does not independently output position, velocity, acceleration, roll, or pitch.

For the current framework:

    yaw_cmd = query-boundary command yaw
    omega_cmd = [0, 0, 0]

Command position, velocity, and yaw at the query boundary are explicit non-learning provenance. This preserves the accepted hover/CEM command state while allowing later contexts to start from their own causal command state.

## One-shot rollout API

The learning-facing API is:

    evaluate_open_loop_batch(
        simulator,
        contexts,
        normalized_actions,
        task,
        record_trajectory=False,
    )

It:

1. validates the frozen model and accepted reward profile;
2. decodes every complete action;
3. constructs all FullState trajectories;
4. deep-clones the corresponding UAV, cable, and residual-history states;
5. advances the batch together on the production GPU simulator;
6. evaluates the existing tuned task reward and hard gates;
7. returns compact GPU-resident metrics.

There is no Python loop over candidate rollouts. `record_trajectory=True` is intentionally restricted to batch size one and provides an optional deterministic debug trajectory.

## One-shot episode result

`OneShotEpisodeResult` returns:

    reward
    feasibility_ranking_reward
    task_success
    feasible
    tip_min_distance_m
    hit_time_s
    directed_tip_speed_m_s
    direction_error_deg
    first_entry_marker
    max_uav_displacement_m
    max_uav_speed_m_s
    max_command_acceleration_m_s2
    rollout_finite
    task_cost

`reward` is the negative of the existing tuned task cost, suitable for a future maximization interface. Hard feasibility and success remain explicit rather than being hidden inside the reward. `feasibility_ranking_reward` exposes the existing feasibility-first CEM ranking score separately.

## Reward reuse

The evaluator imports the existing `VariableWhipAccumulator`. No reward equations were copied into `learning/`.

The required profile is:

    legacy_run_online_strike_margin_tuned_v4

with:

    direction coefficient = 600
    direction shaping target = 20 deg
    directed-speed shaping target = 4.5 m/s

The scientific hard gates remain:

    tip error <= 50 mm
    directed speed >= 4.0 m/s
    direction error <= 30 deg
    c10 first
    UAV displacement <= 0.50 m
    UAV speed <= 3.0 m/s
    acceleration <= 20 m/s^2
    finite rollout

## Policy reward versus model-identification loss

The code now has an explicit boundary:

    learning.one_shot_env
        -> task reward, success, and feasibility

It contains no measured real trajectory, marker residual, parameter-fitting objective, or model-identification loss. Future Real-to-Sim adaptation must use a separate API and cannot be silently inserted into the policy task evaluator.

## Successful CEM reference replay

The existing selected seed-46 artifact was loaded without optimization:

    data/planning_results/
        canonical_whip_variable_duration_tuned_reward_v1/
        2026-08-29T160423.856049Z

Its world-frame knots were transformed once into the yaw-local policy-action frame, normalized, decoded, and passed through the new one-shot evaluator.

| Metric | Saved CEM replay | New interface | Absolute difference |
|---|---:|---:|---:|
| Duration | 1.117534472 s | 1.117534472 s | 0 |
| Hit time | 1.110000014 s | 1.110000014 s | 0 |
| Minimum tip distance | 0.001756333 m | 0.001756298 m | 3.4343e-08 m |
| Directed tip speed | 4.598675728 m/s | 4.598675728 m/s | 0 |
| Direction error | 19.615329742 deg | 19.615329742 deg | 0 |
| First-entry marker | c10 | c10 | identical |
| Maximum UAV displacement | 0.484699547 m | 0.484699547 m | 0 |
| Maximum UAV speed | 2.427251101 m/s | 2.427251101 m/s | 0 |
| Maximum command acceleration | 16.085672379 m/s² | 16.085672379 m/s² | 0 |
| Task result | PASS | PASS | identical |

The only nonzero difference is 0.000034 mm in minimum tip distance, which is below meaningful float32 resolution for this task. The new interface therefore reproduces the authoritative result.

## Tests and verification

Focused Milestone 5A tests:

- root-origin, finite-transform, cable-velocity, direction-normalization, and tensor-schema gate;
- translation and global-yaw invariance;
- normalized 49D action, vector acceleration limit, duration bounds, and reference round-trip;
- FullState integration consistency over the active duration;
- one saved CEM reference replay through the new interface;
- repeated-action batch equivalence between B=1 and B=4.

Results:

    focused Milestone 5A tests:          6 passed
    planning + 4C + 5A focused suite:   22 passed
    complete repository suite:          76 passed
    Python compilation:                 PASS
    git diff --check:                   PASS

The production DDER uses an internal local autograd derivative for bending force. The one-shot outer rollout therefore matches the existing planner's `torch.no_grad()` contract rather than PyTorch `inference_mode`, which would incorrectly disable that internal physical derivative.

## Acceptance criteria

| Criterion | Result |
|---|:---:|
| Structured UAV+cable+goal+theta context | PASS |
| Cable positions and velocities included | PASS |
| Root-centered, yaw-aligned features | PASS |
| One complete variable-duration action | PASS |
| Action dimension exactly 49 | PASS |
| Deterministic consistent FullState generation | PASS |
| Batched fully open-loop evaluation | PASS |
| Existing tuned reward reused | PASS |
| Saved successful CEM action still succeeds | PASS |
| No sequential-feedback policy API | PASS |

## Final summary

    Policy formulation:
        ONE-SHOT OPEN-LOOP

    Context:
        UAV state
        + cable c1...c10 position/velocity
        + target
        + desired direction
        + physics context theta

    Context dimension:
        83

    Action:
        16x3 acceleration knots
        + maneuver duration

    Action dimension:
        49

    Task reward:
        tuned legacy run_online reward

    Successful CEM reference through new interface:
        PASS

    SAC implemented:
        NO

    SAC trained:
        NO

    CEM rerun:
        NO

    Model changed:
        NO

    Protected test:
        NOT EVALUATED

    Real hardware:
        NOT EXECUTED

    MILESTONE5A:
        PASS

## Stop

Milestone 5A is complete. SAC implementation and training have not started.
