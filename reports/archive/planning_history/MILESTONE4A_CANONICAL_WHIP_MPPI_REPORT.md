# Milestone 4A — Canonical Whip Task + Production Full-Horizon MPPI

## 1. Outcome

The single authorized `canonical_whip_v1`, seed-42, simulation-only planning run is **FAIL**. The final decision comes from one deterministic replay of the globally best acceleration-knot trajectory from the exact post-hover state. Failed hard gates: **tip_position, directed_tip_speed, impact_direction, tip_first, uav_displacement, uav_speed**.

No model fitting, residual retraining, physics/geometry change, protected-test evaluation, receding-horizon control, ROS/radio access, or real Crazyflie execution occurred.

## 2. Frozen production model

- Freeze: `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`
- Integrity: verified before planning (13 frozen artifacts re-hashed)
- Predictor: frozen UAV Physics + frozen causal UAV Residual + 12-node DDER
- UAV parameters: K_p=4.020097778647703, K_v=12.05728865003419, k_a=0.7327301468333923, K_R=69.18419375930429, K_omega=11.456583174321011
- Cable parameters: EI=9.225797579462985e-05 N m², Cb=0.0027704569956681496 N m² s
- Geometry: 0.9525 m, attachment [0,0,-0.055] m, c1…c10 at nodes 2…11
- Backend: CUDA float32, PCG32, fused production DDER, 3 substeps, 4 position projections
- Protected test `fig8vertical_002`: **NOT EVALUATED**

## 3. Canonical task

- Initial UAV: [0,0,1.5] m, zero velocity, yaw 0
- Target: [1.0,0.0,1.4] m
- Desired impact direction: [+1,0,0]
- Horizon: 0.70 s; impact window: [0.30,0.70] s
- Tip radius: 0.050 m; directed-speed minimum: 4.0 m/s; direction tolerance: 30°
- Tip-first: c10 must be the first of c1…c10 entering the target sphere
- Hard UAV limits: 0.50 m displacement and 3.0 m/s speed
- Hard command acceleration norm: 20.0 m/s²
- No target collision/contact physics was introduced.

## 4. Initial state and hover pre-roll

A hanging cable was constructed from the exact frozen rest lengths. One 0.50-s production hover pre-roll was run once, producing the common post-hover UAV, cable, DDER, and causal residual-FIFO state cloned into every candidate. The pre-roll final cable-speed RMS was 0.00647537 m/s; the residual FIFO was finite and causally populated. It was not rerun per candidate.

## 5. FullState command parameterization

MPPI optimizes one 11×3 acceleration-knot tensor. Acceleration is linearly interpolated at the 100-Hz production command/physics grid. Velocity is its exact trapezoidal integral; position uses the exact interval integral for linearly varying acceleration. Yaw and body-rate command remain zero. Position, velocity, and acceleration are never independently optimized. Every knot is projected by vector norm, not component clipping.

## 6. Structured nominal

The legacy nominal was audited but belongs to the superseded direct-kinematics/11-node simulator contract. It was not copied into production. The exact fallback profile specified by Milestone 4A was used. Its initial best tip-target distance was 0.3449 m, best-event directed tip speed was 8.203 m/s, cost was 208.527206, and success was False.

## 7. MPPI implementation and cost

The implementation is derivative-free and advances all 2048 candidates together through `build_production_simulator`. It accumulates event/task/safety metrics on GPU and never saves the population trajectories. Candidate 0 is the unperturbed nominal. Stabilized global weights use λ=1.0, perturbation σ=2.5 m/s², and seed 42. The globally best candidate is retained across iterations; only it is replayed in full.

The event objective is the configured minimum over [0.30,0.70] s: 4×normalized position² + 2×directed-speed deficiency² + 1×direction deficiency² + 4×non-tip proximity². Secondary configured weights are 0.02 effort, 0.05 knot smoothness, 0.10 final UAV speed, and 10 each for displacement/speed excess. A valid trajectory receives one finite −20 bonus. Hard gates, not dense cost, define success.

## 8. MPPI configuration

- Horizon: 0.70 s
- Acceleration knots: 11
- Samples: 2048
- Maximum iterations: 4
- Perturbation σ: 2.5 m/s²
- Temperature λ: 1.0
- Seed: 42
- Early stop: success plus at most one polishing iteration

## 9. Iteration history

| Iteration | Current best cost | Best-ever cost | ESS | Min target error [m] | Best directed speed [m/s] | Successes | Runtime [s] |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38.394375 | 38.394375 | 1.00 | 0.0522 | 8.430 | 0 | 0.720 |
| 2 | 25.164829 | 25.164829 | 1.00 | 0.0353 | 8.250 | 0 | 0.772 |
| 3 | 24.285549 | 24.285549 | 4.38 | 0.0265 | 8.479 | 0 | 0.646 |
| 4 | 16.894960 | 16.894960 | 1.60 | 0.0259 | 7.639 | 0 | 0.643 |

## 10. Final deterministic replay

- Valid hit time: none
- Reported event time (hit, otherwise near-miss): 0.700 s
- Tip position error: 56.416 mm
- Tip total speed: 9.634 m/s
- Directed tip speed: 7.808 m/s
- Impact-direction error: 35.860°
- First target-entry marker: none
- Maximum UAV displacement: 1.0939 m
- Maximum UAV speed: 3.9546 m/s
- UAV-to-target distance at event: 0.9794 m
- Tip/UAV speed ratio at event: 2.436
- Maximum command acceleration: 14.3940 m/s²
- Rollout finite: True

The winning sampled row and required batch-one replay were not numerically identical over this aggressive full horizon. The sampled row reported cost 16.894960 and minimum distance 25.936 mm; batch-one replay reported cost 20.576815 and minimum distance 56.416 mm. The pre-run three-step batch check differed by only 1.397e-07, so the full-horizon discrepancy is recorded as amplified float32 batch-shape sensitivity. Neither the sampled population nor the replay contained a hard-gated success, and the replay is authoritative by the frozen protocol.

## 11. Cable-energy diagnostic

At the reported event, observed dynamic cable kinetic energy was 0.25158 J. The distal c8–c10 fraction was 57.54%. Definition: 0.5 m_i||v_i||² over c1…c10, with the distal numerator using c8…c10. This is diagnostic only and did not enter MPPI.

## 12. Runtime

- Total MPPI solve: 2.783 s
- Iterations executed: 4
- Candidate rollouts: 8192
- Effective rollouts/s: 2943.3
- Final deterministic replay: 0.622 s
- Peak CUDA memory: 26.3 MiB
- Five-minute hard stop reached: False

## 13. Verification

Cheap pre-run checks passed: {'command_consistency': {'maximum_velocity_consistency_error': 3.427267074584961e-07, 'maximum_position_consistency_error': 4.842877388000488e-08}, 'projected_maximum_acceleration_m_s2': 19.999998092651367, 'state_clone_independent_storage': True, 'synthetic_task_logic': {'valid': True, 'wrong_direction': False, 'insufficient_speed': False, 'c5_first': False, 'excess_displacement': False}, 'batch_equivalence_maximum_absolute_difference': 1.3969838619232178e-07, 'smoke_candidate_count': 8, 'smoke_cost_finite': True}. Command integration consistency, norm projection, state deep-cloning including residual FIFO, synthetic hard-gate cases, batch equivalence, and a small MPPI smoke solve were checked before the authoritative run.

## 14. Artifacts

- Result directory: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\planning_results\canonical_whip_v1\2026-08-29T041300.459080Z`
- Planner freeze: `not created (simulation failed)`
- FullState CSV/NPZ are marked `SIMULATION_ONLY` and `NOT_AUTHORIZED_FOR_REAL_FLIGHT`.
- The generated command has **NOT** been executed on real hardware.

## 15. Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Task:
        canonical_whip_v1

    Horizon:
        0.70 s

    MPPI:
        samples = 2048
        iterations = 4
        runtime = 2.783 s

    Hit time:
        none

    Tip position error:
        56.416 mm

    Tip total speed:
        9.634 m/s

    Directed tip speed:
        7.808 m/s

    Impact-direction error:
        35.860 deg

    UAV max displacement:
        1.0939 m

    UAV max speed:
        3.9546 m/s

    Tip/UAV speed ratio at hit:
        2.436

    First target-entry marker:
        none

    Max command acceleration:
        14.3940 m/s^2

    MPPI_SIMULATION:
        FAIL

    REAL HARDWARE EXECUTION:
        NOT PERFORMED
