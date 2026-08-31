# Milestone 5B.2 — RL Reward Audit and Focused One-Shot SAC Pilot

## 1. CEM reward versus RL reward

The legacy `legacy_run_online_strike_margin_tuned_v4` reward remains unchanged as the optimizer/reference objective. SAC uses the separate `rl_whip_reward_v1`; no CEM action, elite, or trajectory entered replay or policy training.

## 2. Frozen scientific contract

The model remained `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`. The one-query, fully open-loop 83-D context and 49-D maneuver interface remained unchanged. Scientific success gates, hard feasibility, nominal theta, DDER, geometry, and solver settings were unchanged.

## 3. Exact RL reward

`P=exp(-d^2/(2*0.2^2))`, `V=tanh(v_dir/4.0)`, and direction alignment is smoothly suppressed at zero speed then converges to `0.5*(1+alignment)`. `R_strike=max_t(2P+2PV+PD)`. Total reward is `R_strike + 5*success - 2*(clipped normalized safety violations) - 1*non_tip_first - 0.05*effort - 0.05*smoothness - 0.10*t_hit` for successful episodes. No `/100`, batch normalization, or global clipping is applied.

Each normalized safety violation is independently clipped at 4 only for RL conditioning. Hard feasibility remains unclipped and unchanged.

## 4. Existing-trajectory reward audit

| Trajectory | Success | Feasible | Tip [mm] | Directed [m/s] | Direction [deg] | R_strike | R_success | R_safety | R_non_tip | R_control | R_time | R_RL | Legacy reference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strong_success | True | True | 1.756 | 4.599 | 19.615 | 4.6061 | 5.0000 | -0.0000 | -0.0000 | -0.0354 | -0.1110 | 9.4597 | 599.5731 |
| fragile_success | True | True | 1.103 | 4.282 | 29.983 | 4.5123 | 5.0000 | -0.0000 | -0.0000 | -0.0285 | -0.1090 | 9.3748 | 599.5644 |
| close_wrong_direction | False | True | 5.477 | 4.225 | 47.104 | 4.4072 | 0.0000 | -0.0000 | -0.0000 | -0.0195 | -0.0000 | 4.3877 | -1.1723 |
| close_slow | False | True | 11.654 | 2.002 | 41.035 | 3.7958 | 0.0000 | -0.0000 | -0.0000 | -0.0258 | -0.0000 | 3.7700 | -1.1812 |
| old_aggressive_mppi | False | False | 56.416 | 7.808 | 35.860 | 4.6379 | 0.0000 | -3.0240 | -0.0000 | -0.0261 | -0.0000 | 1.5879 | -20.5768 |
| failed_sac | False | False | 1082.534 | 0.014 | 87.694 | 0.0000 | 0.0000 | -5.2837 | -0.0000 | -0.0220 | -0.0000 | -5.3056 | -51.4155 |
| safe_hover_no_whip | False | True | 1344.502 | 0.009 | 41.828 | 0.0000 | 0.0000 | -0.0000 | -0.0000 | -0.0000 | -0.0000 | 0.0000 | -39.4397 |
| bad_random_00 | False | False | 923.958 | 0.017 | 86.611 | 0.0001 | 0.0000 | -16.0000 | -0.0000 | -0.1233 | -0.0000 | -16.1233 | -1901458.3750 |

All 23 audit rows, including every random rollout, are stored in `reward_audit_results.json`.

## 5. Analytic reward slices

Distance, directed-speed, direction, displacement, and speed slices were finite and were checked for the expected monotonic behavior. The saved plots are `reward_distance_curve.png`, `reward_speed_curves.png`, `reward_direction_curves.png`, and `reward_safety_curves.png`.

## 6. Reward-audit gate

**REWARD_AUDIT: PASS**

Observed audit reward range: [-16.123283, 9.459671].

## 7. Focused SAC pilot

The pilot used the canonical settled state plus the 128 closest states from the existing 5B training bank. Validation used the canonical state and 32 closest states from the separate existing validation bank. Targets were 50% exact canonical and 50% sampled from x=[0.95,1.05], y=[-0.05,0.05], and canonical local z +/-0.03 m.

The actor and twin critics remained 256-256-256 SiLU networks. The radial stochastic transform and one-terminal-decision target remained unchanged. Critic regression used SmoothL1/Huber with delta 1.0. Training began after 10,240 actor-generated episodes, used eight updates per 2,048 rollouts, and used no demonstrations.

## 8. Pilot progression and final result

Untrained canonical tip error: 1347.748 mm; final best-checkpoint canonical tip error: 1022.851 mm.

Untrained held-out success/feasibility: 0.00% / 100.00%. Final held-out success/feasibility: 0.00% / 0.00%.

Collected episodes: 301056; gradient updates: 1144; training runtime: 255.806 s.

Pilot classification: **SAC_STILL_EXPLORATION_LIMITED**.

### Validation progression

| Episodes | Canonical success | Canonical tip [mm] | Held success | Held feasible | Held tip [mm] | Held directed [m/s] | Held direction [deg] | Duration [s] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0% | 1347.7 | 0.0% | 100.0% | 1331.4 | 0.137 | 15.82 | 0.825 |
| 20,480 | 0.0% | 1245.1 | 0.0% | 100.0% | 1259.8 | 0.691 | 60.48 | 0.756 |
| 40,960 | 0.0% | 1022.9 | 0.0% | 0.0% | 1002.6 | 0.149 | 69.92 | 0.567 |
| 61,440 | 0.0% | 1143.3 | 0.0% | 100.0% | 1142.3 | 0.289 | 52.03 | 0.487 |
| 81,920 | 0.0% | 1351.4 | 0.0% | 100.0% | 1348.3 | 0.090 | 76.32 | 0.469 |
| 100,352 | 0.0% | 1351.3 | 0.0% | 100.0% | 1347.7 | 0.154 | 51.34 | 0.462 |
| 120,832 | 0.0% | 1351.1 | 0.0% | 100.0% | 1334.8 | 0.150 | 41.33 | 0.460 |
| 141,312 | 0.0% | 1344.0 | 0.0% | 100.0% | 1328.4 | 0.192 | 43.48 | 0.461 |
| 161,792 | 0.0% | 1322.4 | 0.0% | 100.0% | 1308.9 | 0.168 | 44.13 | 0.461 |
| 180,224 | 0.0% | 1320.2 | 0.0% | 100.0% | 1301.5 | 0.170 | 43.62 | 0.460 |
| 200,704 | 0.0% | 1323.1 | 0.0% | 100.0% | 1304.8 | 0.155 | 33.36 | 0.461 |
| 221,184 | 0.0% | 1314.7 | 0.0% | 100.0% | 1299.9 | 0.225 | 37.33 | 0.461 |
| 241,664 | 0.0% | 1311.5 | 0.0% | 100.0% | 1292.6 | 0.194 | 40.52 | 0.461 |
| 260,096 | 0.0% | 1329.1 | 0.0% | 100.0% | 1313.6 | 0.174 | 39.50 | 0.461 |
| 280,576 | 0.0% | 1331.7 | 0.0% | 100.0% | 1314.5 | 0.186 | 32.09 | 0.462 |
| 301,056 | 0.0% | 1322.1 | 0.0% | 100.0% | 1312.0 | 0.162 | 36.17 | 0.461 |

The stored best checkpoint is the 40,960-episode checkpoint because all checkpoints tied at zero success and it had the smallest held-out tip distance. It is infeasible and is not an accepted policy. The endpoint checkpoint returned to 100% feasibility but remained a safe no-whip mode with 1312.0-mm held-out median tip error.

### Final canonical hard-gate result

The selected diagnostic checkpoint produced:

- tip error: 1019.213 mm in authoritative single-row replay;
- directed tip speed: 0.010 m/s;
- direction error: 57.503 deg;
- tip-first: false;
- maximum UAV displacement: 0.802 m;
- maximum UAV speed: 2.638 m/s;
- maximum command acceleration: 7.210 m/s²;
- duration: 0.562 s;
- finite rollout: true.

It failed tip position, directed speed, direction, tip-first, and UAV-displacement gates. UAV speed, command acceleration, and numerical finiteness passed.

### Training numerics and pathology

The RL-native reward fixed the critic-conditioning failure:

- observed audit-plus-training reward range: [-16.127930, 9.459671];
- Q1 Huber loss range: [1.3829, 11.4387];
- Q2 Huber loss range: [1.3805, 11.5050];
- maximum recorded unclipped critic gradient norms: 19.31 and 14.51;
- alpha range: [0.7122, 0.9987];
- estimated entropy range: [19.82, 22.96];
- mean normalized acceleration-knot norm range: [0.697, 0.780].

There was no critic explosion, NaN/Inf, or entropy collapse. Nevertheless, duration fell from approximately 0.825 s before training to approximately 0.461 s at the endpoint, close to the 0.45-s lower envelope. The actor learned a short, low-reward, safe no-whip mode rather than discovering the sparse strike region. The pilot therefore failed the defined `PROMISING` test as well as hard success: target-distance improvement was not accompanied by the required directed-speed and direction improvement while retaining healthy feasibility.

## 9. Comparison with failed 5B

Milestone 5B used the legacy optimizer reward divided by 100 and produced critic losses up to approximately 1e15 with zero success. This pilot uses the separately bounded RL reward and Huber critics. The comparison is methodological; the model, policy interface, and hard task gates remain fixed.

The main scientific distinction is now clear: reward numerical conditioning improved dramatically, but pure one-shot stochastic exploration still did not locate a whip under the retained variable-duration 49-D interface.

## 10. Safety boundary

CEM training data: **NOT USED**. Physics conditioning: **NOT ENABLED**. Protected test: **NOT EVALUATED**. Real hardware: **NOT EXECUTED**.

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Policy:
        One-Shot Terminal SAC

    Execution:
        OPEN LOOP

    Scientific success gates:
        UNCHANGED

    CEM training data:
        NOT USED

    Optimizer/reference reward:
        legacy_run_online_strike_margin_tuned_v4

    SAC reward:
        rl_whip_reward_v1

    RL reward observed range:
        [-16.127930, 9.459671]

    Reward audit:
        PASS

    SAC pilot episodes:
        301056

    Canonical:
        tip error = 1019.213 mm
        directed speed = 0.010 m/s
        direction error = 57.502 deg
        feasible = False
        result = FAIL

    Near-canonical held-out:
        success = 0.00 %
        feasible = 0.00 %

    SAC pilot:
        SAC_STILL_EXPLORATION_LIMITED

    Physics conditioning:
        NOT ENABLED

    Protected test:
        NOT EVALUATED

    Real hardware:
        NOT EXECUTED
