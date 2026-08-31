# Milestone 4C — Variable-Duration Offline CEM Report

## Outcome

Variable-duration CEM was implemented and evaluated against the unchanged `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`. The prior fixed 0.70-s formulation was replaced because Milestone 4B.1's best feasible event occurred at that artificial boundary. CEM jointly optimized 16 normalized-time acceleration knots and maneuver duration.

Objective profile: **legacy `run_online` strike-reward port under the current CEM/task contract**.

The physical model, UAV residual, cable parameters, geometry, PCG32 backend, float32 precision, three DDER substeps, and four projections were not changed or refitted. The protected test was not evaluated and no real hardware was connected or executed.


## Legacy `run_online` reward used

The reward kernel and active corrected GUI-preset coefficients were read from
`legacy/current_baseline_2026-08-27/drone_mpc/mppi.py` and
`legacy/current_baseline_2026-08-27/drone_mpc/receding_mppi_gui.py`; they were
not reconstructed from a report.  At each candidate event the port evaluates:

    J_position = 40 d^2 / (d^2 + 0.18^2)
    proximity = exp(-d^2 / (2 * 0.12^2))
    J_speed = 25 proximity relu(v_min - v_directed)^2
    predictive_proximity = exp(-d^2 / (2 * 0.45^2))
    J_predictive = 10 predictive_proximity relu(0.25 v_min - v_directed)^2
    J_direction = 20 proximity relu(cos(theta_max) - cos(theta))^2
    J_displacement = 2 ||p_uav(event) - p_uav(0)||^2

It also uses safety weight 180, success bonus
`-600`, effort weight 1e-05, and
smoothness weight 1e-05.  The predictive-speed
weight of 10 is the value initialized and
applied by the corrected legacy `run_online` GUI preset.

Scientific separation was preserved: this run did **not** restore the legacy
simulator, MPPI algorithm, or legacy 3.5-m/s/35-degree acceptance defaults.
It used the frozen production model, variable-duration CEM, feasibility-first
ordering, and the current hard gates of 4.0 m/s and 30 degrees.  Event and
safety bookkeeping use the current fixed-timestep production planner contract;
therefore this is a faithful reward-shape/coefficient port, not a byte-for-byte
execution of the complete legacy `_mppi_event_objective` implementation.


## Formulation and implementation

- Decision dimension: 49 (48 acceleration components + one duration).
- Initial duration range: 0.45–1.20 s.
- One-time extension used: **NO**.
- Final allowed duration maximum: 1.20 s.
- Fixed physics timestep; candidate-specific `t <= T_i` masks exclude all post-duration states.
- A candidate terminates scientifically at its first valid strike.
- FullState position and velocity are integrated from linearly interpolated acceleration; yaw and omega commands remain zero.
- Population: 8192, evaluated as four canonical 2048-row chunks.
- Elite fraction: 5% (~410 candidates).
- Covariance: full 49×49 with positive-definite regularization and configured variance floors.
- Distribution smoothing: 30% old + 70% elite.
- Feasibility-first global elite ordering: successes, feasible near-misses, then infeasible candidates with continuous violation ordering.
- Every completed iteration was checkpointed with distribution, RNG state, best candidate, bounds, model reference, and iteration summary.

## Cheap pre-run verification

```json
{
  "batch_consistency_tolerances_m": {
    "final_c10_position": 0.002,
    "final_uav_position": 0.001,
    "minimum_tip_target_distance": 0.002
  },
  "checks": {
    "b1_b8_b2048_full_horizon_numerical_contract": "PASS",
    "cem_sampling_and_covariance_update_finite": "PASS",
    "checkpoint_distribution_rng_and_best_candidate_roundtrip": "PASS",
    "duration_and_vector_acceleration_projection": "PASS",
    "feasibility_first_elite_ordering": "PASS",
    "first_valid_success_terminates_scientific_evaluation": "PASS",
    "global_8192_candidate_elite_order_across_2048_boundaries": "PASS",
    "one_iteration_2048_row_production_cuda_cem_smoke": "PASS (1.286 s)",
    "post_duration_states_excluded_from_cost_and_feasibility": "PASS",
    "variable_duration_command_integration_consistency": "PASS"
  },
  "fixed_uav_evaluation_batch_size": 2048,
  "pass": true,
  "protected_test_evaluated": false,
  "pytest_command": ".venv\\Scripts\\python.exe -m pytest tests\\test_milestone4c_variable_cem.py -q",
  "pytest_result": "6 passed",
  "real_hardware_executed": false,
  "schema": "milestone4c_preflight_verification_v1"
}
```

## Per-seed campaign

| Seed | Iterations | Success | Tip error [m] | Hit [s] | T [s] | Direction [deg] | Directed speed [m/s] | UAV disp. [m] | UAV speed [m/s] | Runtime [s] |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 30 | False | 0.0133 | — | 0.981 | 52.63 | 4.021 | 0.497 | 2.079 | 159.6 |
| 43 | 30 | False | 0.0055 | — | 1.003 | 51.99 | 3.943 | 0.488 | 2.007 | 158.9 |
| 44 | 30 | False | 0.0125 | — | 0.997 | 51.29 | 4.053 | 0.495 | 2.199 | 158.2 |
| 45 | 30 | False | 0.0139 | — | 0.985 | 51.42 | 4.083 | 0.491 | 2.131 | 158.6 |
| 46 | 30 | False | 0.0055 | — | 0.994 | 47.10 | 4.225 | 0.493 | 1.876 | 157.4 |
| 47 | 30 | False | 0.0126 | — | 0.998 | 50.04 | 3.981 | 0.496 | 1.970 | 158.3 |

Final seed: **46**. Selection class: **best_feasible_near_miss**. Successful feasible plans, when present, were ranked by target-center accuracy, then direction, then feasibility margin; duration was only a weak tie-breaker.

## Deterministic replay and numerical consistency

- Optimized duration: 0.9945 s (INTERIOR).
- First valid hit: NONE.
- Tip error: 5.477 mm.
- Tip total speed: 6.208 m/s.
- Directed tip speed: 4.225 m/s.
- Direction error: 47.104 deg.
- First target-entry marker: c10.
- UAV maximum displacement: 0.4930 m.
- UAV maximum speed: 1.8759 m/s.
- Maximum command acceleration: 12.1380 m/s².
- Tip/UAV speed ratio at event: 11.878.
- Candidate/replay consistency: **PASS**.

| Hard gate | Result |
|---|:---:|
| command_acceleration | PASS |
| directed_tip_speed | PASS |
| finite | PASS |
| impact_direction | FAIL |
| tip_first | PASS |
| tip_position | PASS |
| uav_displacement | PASS |
| uav_speed | PASS |

If the hit is later than 1.0 s, the artifact is explicitly labeled `LONGER_THAN_PRIMARY_VALIDATION_HORIZON`; it is not rejected in simulation solely for that reason.

## Figure-8 source audit

The bounded broader audit status is **NOT FOUND**. Figure-8 Task B was **NOT RUN** because an authoritative current generator and unambiguous endpoint convention were not found. See `reports/figure8_source_audit_4c.json`.

## Runtime and artifacts

- Campaign runtime: 952.88 s.
- Candidate rollouts: 1474560.
- Effective rollouts/s: 1547.5.
- Peak CUDA memory: 41.8 MB.
- Result directory: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\planning_results\canonical_whip_variable_duration_legacy_reward_v1\2026-08-29T144858.075926Z`.
- Video: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\planning_results\canonical_whip_variable_duration_legacy_reward_v1\2026-08-29T144858.075926Z\canonical_whip_variable_duration_legacy_reward_v1_final_replay.mp4`.
- Email delivery: `NOT_AVAILABLE_IN_LOCAL_ENVIRONMENT`.
- Protected test: `NOT EVALUATED`.
- Real hardware: `NOT EXECUTED`.

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Optimizer:
        Variable-Duration CEM

    Canonical task:
        target = [1.0, 0.0, 1.4]

    Duration search:
        initial range = [0.45, 1.20] s
        extended = NO
        final allowed max = 1.20 s

    Seeds executed:
        [42, 43, 44, 45, 46, 47]

    Final selected seed:
        46

    Optimized maneuver duration:
        0.9945 s

    Hit time:
        NONE

    Duration status:
        INTERIOR

    Tip error:
        5.477 mm

    Tip total speed:
        6.208 m/s

    Directed tip speed:
        4.225 m/s

    Direction error:
        47.104 deg

    First target-entry marker:
        c10

    UAV max displacement:
        0.4930 m

    UAV max speed:
        1.8759 m/s

    Max command acceleration:
        12.1380 m/s^2

    Candidate/replay consistency:
        PASS

    VARIABLE_DURATION_CEM:
        FAIL

    Figure-8 source:
        NOT FOUND

    Figure-8 task:
        NOT RUN

    Video:
        C:\Users\wts28\Documents\PHD\particle_filter_cable_project\data\planning_results\canonical_whip_variable_duration_legacy_reward_v1\2026-08-29T144858.075926Z\canonical_whip_variable_duration_legacy_reward_v1_final_replay.mp4

    Protected test:
        NOT EVALUATED

    Real hardware:
        NOT EXECUTED
