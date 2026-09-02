# PPO Feedback Dependence and Open-Loop Trajectory-Compiler Report

## 1. Question and frozen scope

This zero-training milestone asks whether the terminal sequential PPO must remain a 10-Hz cable-state-feedback controller, or whether one initial measurement plus a fast PPO+sim rollout can compile a robust high-level open-loop FullState command sequence. The policy checkpoint, nominal model, canonical target, reward, 83-D context, 6-D control action, 10.0-s horizon, and success definition were frozen. No CEM, learning, target randomization, theta training, protected data, or hardware was used.

Checkpoint: `data/policy_training/whip_ppo_directional_displacement15_v1/2026-08-31T104735.951178Z/checkpoints/latest.pt` at 1,001,472 episodes. This is the requested terminal checkpoint, not the best-validation checkpoint.

## 2. What was compiled

The existing PPO is queried 100 times at 0.1-s intervals. The production UAV/residual/12-node DDER simulator advances 1,000 times at 0.01 s. At every physics step, after the sequential controller has resolved its live-state re-anchoring, the compiler records:

`[p_cmd, v_cmd, a_cmd, q_cmd, omega_cmd]`.

The saved trajectory is therefore 1,000 physical FullState commands. Its replay API accepts no policy, context, or cable measurement. It is high-level open loop; the frozen low-level FullState UAV tracking dynamics remain part of the physical plant/controller contract.

## 3. Exact same-state replay gate

The normal feedback PPO rollout and compiled replay used all 512 bank states and identical deterministic float32 production physics. Maximum trajectory differences were:

```json
{
  "uav_position_m": 0.0,
  "uav_velocity_m_s": 0.0,
  "uav_orientation_xyzw": 0.0,
  "uav_angular_velocity_world_rad_s": 0.0,
  "cable_positions_m": 0.0,
  "cable_velocities_m_s": 0.0
}
```

Hard classification/metric differences were:

```json
{
  "task_success": 0,
  "endpoint_success": 0,
  "legacy_scientific_success": 0,
  "finite": 0,
  "first_entry_marker": 0,
  "first_entry_physics_step": 0,
  "first_entry_time_s": 0.0,
  "first_entry_tip_distance_m": 0.0,
  "first_entry_tip_speed_m_s": 0.0,
  "first_entry_directed_speed_m_s": 0.0,
  "first_entry_direction_error_deg": 0.0,
  "minimum_tip_distance_m": 0.0,
  "maximum_uav_displacement_m": 0.0,
  "maximum_uav_speed_m_s": 0.0,
  "maximum_command_acceleration_m_s2": 0.0
}
```

Gate: **PASS**. The mismatch and disturbance experiments were run only because this gate passed.

## 4. Nominal 512-state architecture controls

| Mode | Task success | Endpoint success | Finite | Median max UAV displacement |
|---|---:|---:|---:|---:|
| Feedback PPO | 91.60% | 99.61% | 100.00% | 0.6727 m |
| Matched compiled open loop | 91.60% | 99.61% | 100.00% | 0.6727 m |
| Wrong-state compiled sequence | 15.43% | 17.58% | 94.92% | 0.6852 m |

Matched compilation is expected to equal feedback under deterministic nominal physics. The wrong-state control isolates whether measuring/compiling for the actual initial state matters.

The wrong-state replay also reproduced the previously diagnosed FullState tracking-runaway failure mode: its p95 maximum displacement was `6.059e16 m` and p95 UAV speed was `3.376e17 m/s`. Those absurd-but-finite post-failure values are not interpreted as physical exploration or task behavior. They strengthen the conclusion that a command sequence compiled for the wrong initial condition is unsafe to reuse.

## 5. EI/Cb model-mismatch sweep

PPO observations continued to contain nominal theta. Compilation was always performed under nominal theta; only execution physics changed.

| Case | EI scale | Cb scale | Feedback | Compiled open loop | Feedback - compiled |
|---|---:|---:|---:|---:|---:|
| EI_0p80 | 0.80 | 1.00 | 92.58% | 72.66% | +19.92 pp |
| EI_0p90 | 0.90 | 1.00 | 92.97% | 73.44% | +19.53 pp |
| EI_1p10 | 1.10 | 1.00 | 91.80% | 74.41% | +17.38 pp |
| EI_1p20 | 1.20 | 1.00 | 91.02% | 74.80% | +16.21 pp |
| Cb_0p80 | 1.00 | 0.80 | 91.41% | 75.39% | +16.02 pp |
| Cb_0p90 | 1.00 | 0.90 | 93.36% | 77.15% | +16.21 pp |
| Cb_1p10 | 1.00 | 1.10 | 90.04% | 70.90% | +19.14 pp |
| Cb_1p20 | 1.00 | 1.20 | 88.48% | 63.28% | +25.20 pp |
| EI_Cb_0p80 | 0.80 | 0.80 | 91.60% | 77.15% | +14.45 pp |
| EI_Cb_1p20 | 1.20 | 1.20 | 87.70% | 65.62% | +22.07 pp |

## 6. Post-planning disturbances

All impulses were applied after physics step 150 (1.50 s), after compilation had observed x0. Feedback PPO could react at subsequent 10-Hz queries; compiled replay could not.

| Disturbance | Step | Feedback | Compiled open loop | Feedback - compiled |
|---|---:|---:|---:|---:|
| uav_lateral_velocity_0p15 | 150 | 90.62% | 64.06% | +26.56 pp |
| distal_cable_lateral_velocity_0p25 | 150 | 93.36% | 82.62% | +10.74 pp |
| combined_lateral_impulse | 150 | 91.60% | 76.56% | +15.04 pp |

## 7. Compilation latency

Batch-one end-to-end timing includes 100 context builds, 100 PPO queries, 1,000 production UAV/residual/DDER steps, and command recording.

| Numerical path | Median | P95 | Max |
|---|---:|---:|---:|
| Logical B=1 residual evaluation | 12170.98 ms | 12362.43 ms | 12384.86 ms |
| Residual padded to fixed 2048 | 12825.97 ms | 12995.74 ms | 13009.81 ms |

The optimized B=1 timing is the relevant deployment compiler latency. The fixed-2048 result is retained to expose the numerical-padding overhead; neither path changes the model equations.

## 8. Architecture decision

Decision: **TEN_HZ_CLOSED_LOOP_PPO**.

Reason: optimized compilation p95 12.362 s exceeds 1.0 s; mean mismatch feedback advantage 18.61 pp exceeds 10 pp; worst mismatch feedback advantage 25.20 pp exceeds 20 pp; mean disturbance feedback advantage 17.45 pp exceeds 10 pp

The decision rule was fixed in the audit artifact: compilation must replay exactly, optimized B=1 p95 must be <=1.0 s, the mean feedback advantage across EI/Cb cases must be <=10 percentage points, the worst mismatch advantage <=20 points, and the mean disturbance advantage <=10 points. This is an architecture-screening rule, not a scientific task gate.

## 9. What did not happen

- PPO training: **NONE**
- CEM: **NOT USED**
- Target generalization: **NOT TESTED**
- Theta-conditioned policy learning: **NOT ENABLED**
- Displacement retraining: **NOT PERFORMED** (displacement was recorded)
- Protected `fig8vertical_002`: **NOT EVALUATED**
- Real hardware: **NOT EXECUTED**

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Policy:
        TERMINAL SEQUENTIAL PPO (1,001,472 episodes)

    New training:
        NONE

    State-bank contexts:
        512

    Actor queries during compilation:
        100

    Recorded FullState commands:
        1000

    Exact replay:
        PASS

    Nominal feedback success:
        91.60%

    Nominal compiled success:
        91.60%

    Wrong-state compiled success:
        15.43%

    Mean feedback advantage under EI/Cb mismatch:
        +18.61 percentage points

    Mean feedback advantage under post-t0 disturbance:
        +17.45 percentage points

    Optimized compilation median:
        12170.98 ms

    Optimized compilation p95:
        12362.43 ms

    Preferred architecture:
        TEN_HZ_CLOSED_LOOP_PPO

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
