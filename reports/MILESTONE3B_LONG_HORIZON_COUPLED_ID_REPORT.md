# Milestone 3B — Long-Horizon Coupled Identification Report

**Repository:** `C:\Users\wts28\Documents\PHD\particle_filter_cable_project`  
**Date:** 2026-08-28  
**Model version:** `aerial_cable_attitude_coupled_v1`  
**Identification methodology:** `full_episode_constrained_multiple_shooting_v1`  
**Final status:** **STOPPED AT THE SYNTHETIC INVERSE-RECOVERY GATE**

## Executive outcome

The model/episode/multiple-shooting structure was implemented and its eight required structural tests pass. However, the required synthetic inverse-recovery test did not complete under the predeclared augmented-Lagrangian schedule. The second finite run was terminated after approximately 65 minutes 54 seconds because the 800-update schedule was computationally impractical even for one three-second synthetic episode.

The scientific stop condition was therefore applied.

- Real seven-parameter physical fitting: **NOT RUN**.
- Physical-only provisional predictive validation: **NOT RUN**.
- Long-horizon residual refit: **NOT RUN**.
- Physics-plus-residual validation: **NOT RUN**.
- `fig8vertical_002` predictive evaluation: **NOT RUN**.
- MPPI: **NOT RUN**.
- Long-horizon pre-test freeze: **NOT CREATED**.

This is not a successful Milestone 3B identification result. It is a structurally verified but computationally unvalidated inverse pipeline that stopped before touching the real-data fit.

## A. Model revision

### Historical direct-acceleration model

The Milestone 3A model formed:

```text
a_ctrl = K_p (p_cmd - p) + K_v (v_cmd - v) + k_a a_cmd

p_dot = v
v_dot = a_ctrl [+ residual]
```

The same nominal acceleration generated desired roll/pitch and yaw heading for the attitude response, but actual attitude did not affect realized translation.

That historical numerical path remains explicitly available as:

```text
translation_mode = direct_effective
model_version = historical_window_direct_effective_v1
```

Historical Stage A, delay, residual-transfer, and frozen untouched-evaluation code paths explicitly select it. Old artifacts were not rewritten.

### New attitude/thrust-coupled model

The new production mode forms:

```text
e_p = p_cmd - p
e_v = v_cmd - v

a_ctrl = K_p e_p + K_v e_v + k_a a_cmd

f_des = a_ctrl + g e_z
s_des = ||f_des||
b3_des = f_des / ||f_des||
R_des = geometric_attitude(b3_des, yaw_cmd)
```

Attitude response remains:

```text
omega_dot_body = K_R e_R(R_des, R) + K_omega (omega_cmd_body - omega_body)
```

For current data `omega_cmd = 0`. Realized physical/effective acceleration is:

```text
b3_actual = R_actual e_z_body
a_real_phys = s_des b3_actual - g e_z
```

When the optional causal residual is enabled later, placement is:

```text
a_real = a_real_phys + Delta_a
```

The residual does not alter `a_ctrl`, `f_des`, `R_des`, yaw demand, or attitude response.

### Exact integration ordering

The implemented one-step order is:

```text
current UAV state + current FullState command
    -> controller acceleration demand a_ctrl
    -> desired attitude R_des from a_ctrl + gravity + commanded yaw
    -> current body angular velocity
    -> body angular acceleration
    -> semi-implicit next body angular velocity
    -> normalized next quaternion R_{k+1}
    -> actual next body-z direction
    -> realized thrust/gravity acceleration
    -> optional realized-translation residual
    -> semi-implicit next velocity
    -> next position
    -> rigid attachment/clamped tangent from that exact next UAV pose
    -> production DDER step
```

Translation over `k -> k+1` therefore uses the semi-implicitly updated `R_{k+1}`. The same `R_{k+1}` drives realized UAV acceleration, attachment-offset rotation, and the clamped cable exit tangent.

The required identity passes: when `R = R_des` and angular velocity is zero, `a_real_phys = a_ctrl` to numerical precision.

## B. Physical-episode methodology

A new `PhysicalEpisode` object represents one maximally long continuously propagatable part of a physical take. It contains take identity, source indices, Motive time, FullState commands, measured UAV/cable outputs, independent observation masks, one causal initialization history, residual-compatible start metadata, and provenance.

An episode transition is propagatable only when:

- adjacent Motive times are finite and strictly increasing;
- the current FullState command is valid and finite;
- no explicit Episode Break occurs between the two samples.

Command gaps split episodes. No command is interpolated or fabricated. Short but genuine command-valid fragments remain visible as episodes; fragments that cannot produce even one step after causal initialization are rejected explicitly.

Measurement quality does not split propagation. UAV invalid/jump flags and cable marker/dropout/jump/geometry flags create observation masks. Manual Use/Exclude annotations now control loss inclusion only. A separate `episode_breaks_s` annotation records true physical discontinuities.

Each episode receives exactly one measured initialization using the existing causal initializer:

- UAV position/orientation at episode start;
- five-frame causal UAV velocity and angular-velocity estimates;
- rigid clamped nodes 0/1;
- measured cable markers at nodes 2,4,...,20;
- midpoint latent odd nodes;
- causal marker velocities;
- production position/velocity projection.

Measurements after episode start are loss targets only. They never overwrite UAV state, cable state, shooting state, or residual FIFO.

### Frozen dataset roles

| Role | Takes |
|---|---|
| Training | `osc_001`, `fig8_001`, `fig8_002`, `fig8vertical_001`, `osc_002` |
| Provisional Validation | `fig8_003`, `osc_003` |
| Protected Test | `fig8vertical_002` |

`fig8vertical_002` was already processed before this milestone. The valid claim is that it was protected from fitting, selection, normalization, solver decisions, and predictive evaluation.

### Episode counts

| Take | Role | Episodes | Propagatable duration [s] | 1-s shooting intervals | UAV observations | Cable-marker observations |
|---|---|---:|---:|---:|---:|---:|
| `fig8_001` | Training | 1 | 40.49 | 41 | 4,050 | 40,500 |
| `fig8_002` | Training | 3 | 27.84 | 30 | 2,787 | 27,870 |
| `fig8vertical_001` | Training | 10 | 67.44 | 74 | 6,754 | 67,540 |
| `osc_001` | Training | 2 | 17.86 | 19 | 1,788 | 17,876 |
| `osc_002` | Training | 5 | 34.41 | 38 | 3,446 | 34,460 |
| **Training total** | 5 takes | **21** | **188.04** | **202** | **18,825** | **188,246** |
| `fig8_003` | Validation | 2 | 56.41 | 58 | 5,643 | 55,546 |
| `osc_003` | Validation | 1 | 15.50 | 16 | 1,551 | 15,510 |
| **Validation total** | 2 takes | **3** | **71.91** | **74** | **7,194** | **71,056** |

The former 320 Training and 184 Validation residual-window counts remain historical short-horizon diagnostics. They are not authoritative Milestone 3B fitting units.

## C. Multiple shooting

### Shooting state

Every internal boundary carries:

- UAV position, velocity, quaternion, and world-frame angular velocity;
- cable positions and velocities for dynamic nodes 2 through 20;
- residual FIFO only when a future residual fit is enabled.

Clamped nodes 0 and 1 are not independent variables. They are derived from UAV pose and measured attachment geometry.

The physical-only raw state has 127 scalar variables per internal boundary: 13 UAV/quaternion values plus 114 cable dynamic-node values. The continuity defect uses 126 values because orientation is represented by a three-vector SO(3) logarithm rather than four quaternion-component differences.

The Training dataset has 202 shooting intervals and 21 episodes, hence 181 internal boundaries, approximately 22,987 raw shooting variables, and 22,806 normalized equality constraints in addition to seven physical parameters.

### Retraction

Every candidate boundary is retracted without measurements:

1. normalize quaternion;
2. derive clamped nodes 0/1 from UAV state;
3. insert candidate dynamic cable nodes;
4. apply the production cable position projection;
5. apply the production velocity projection.

Autograd through retraction is finite. Edge lengths, clamped boundary, and velocity constraints pass the structural test.

### Continuity defects

The implemented blocks are:

- UAV position and velocity Euclidean defects;
- shortest SO(3) log-vector orientation defect;
- angular-velocity Euclidean defect;
- dynamic cable position and velocity defects;
- residual FIFO defect when applicable.

Fixed scales are:

| Block | Scale |
|---|---:|
| UAV position | 0.1 m |
| UAV velocity | 1.0 m/s |
| Orientation | 0.1 rad |
| Angular velocity | 1.0 rad/s |
| Cable position | 0.1 m |
| Cable velocity | 1.0 m/s |
| Residual FIFO feature | 1.0 |

Configured acceptance remains RMS `<= 1e-5` and maximum `<= 1e-4`.

The production four-iteration cable projection is not perfectly idempotent. Reprojecting an exact production trajectory state produced a small normalized continuity floor of approximately `4.28e-6` RMS in the structural test. This is inside the frozen tolerance and was not hidden or relaxed.

### Augmented-Lagrangian implementation

The implemented objective is:

```text
L_A = L_observation
    + mean(lambda * c)
    + rho/2 * mean(c^2)
```

Primal variables are seven smoothly log-bounded physical parameters plus complete internal shooting states. Multipliers and penalty are updated explicitly after each outer loop. The predeclared schedule was:

- 8 outer iterations;
- 100 Adam inner updates per outer iteration;
- learning rate 0.005;
- initial penalty 1;
- penalty multiplier 10;
- maximum penalty 1,000,000;
- gradient clipping 10.

This solver structure is implemented but **not validated**, because the required synthetic recovery did not complete.

## D. Structural verification

All eight required structural tests pass:

| Test | Result |
|---|---|
| 1. Perfect attitude implies realized acceleration equals controller demand | PASS |
| 2. Perturbing actual attitude changes realized acceleration | PASS |
| 3. Changing attitude response gains changes translation | PASS |
| 4. One actual orientation drives both translation and cable clamp | PASS |
| 5. Exact-trajectory shooting states satisfy configured continuity tolerance | PASS |
| 6. Exact-continuity multiple shooting matches single shooting | PASS |
| 7. Interior observation perturbation changes loss but not shooting initialization | PASS |
| 8. DDER retraction is constrained and differentiable | PASS |

One numerical correction was required during this verification. The historical quaternion geodesic used `acos(clamp(dot, max=1-1e-12))`. In float32, `1-1e-12` rounds to 1, causing an infinite derivative at nearly identical orientations. The long-horizon optimizer now evaluates the mathematically equivalent shortest-quaternion SO(3) angle with `atan2`. This changes no model equation or orientation metric; it makes the same metric differentiable in both float32 and float64.

## E. Synthetic inverse-recovery gate

### Synthetic problem

One deterministic three-second coupled episode was generated with the production attitude-coupled UAV, rigid clamp, and production DDER. It contains a 0.20-s initial hover followed by three-axis sinusoidal position/velocity/acceleration excitation and yaw excitation.

| Parameter | True | Initial |
|---|---:|---:|
| `K_p` | 4.0 | 3.0 |
| `K_v` | 5.0 | 6.0 |
| `k_a` | 1.10 | 0.88 |
| `K_R` | 60.0 | 45.0 |
| `K_omega` | 12.0 | 15.0 |
| `EI` | 2.4e-6 | 3.6e-6 |
| `Cb` | 4.2e-8 | 2.52e-8 |

All values are interior to the existing bounds. The episode builder produced one 2.96-s fitting episode after the five-frame causal initialization requirement, split into three numerical shooting intervals.

### Attempt 1

Attempt 1 stopped after approximately 113 seconds when the optimizer produced a non-finite EI variable. Audit traced this to the float32 `acos` orientation-loss derivative described above. The orientation evaluation was corrected to the equivalent stable `atan2` form, and all structural tests were rerun successfully.

### Attempt 2

Attempt 2 remained finite but did not complete the predeclared 800 inner updates. It ran from approximately 11:47:33 to 12:53:27 local time: about **3,954 s (65 min 54 s)**. At a measured intermediate point it consumed approximately 17 GB of private process memory. The process was responsive and computing, but the schedule extrapolated to many hours for this three-second episode.

The run was terminated as computationally impractical. Because the runner emitted only a final result, no valid final recovered-parameter or continuity result exists.

### Gate verdict

| Requirement | Result |
|---|---|
| Finite implementation after SO(3) correction | Observed during partial run |
| Decreasing complete optimization trace | Not available |
| Continuity convergence | **NOT ESTABLISHED** |
| Seven-parameter recovery | **NOT ESTABLISHED** |
| Synthetic gate passed | **NO** |

The instruction states that real data must not be fitted unless this gate passes. That stop was obeyed.

## F. Real physical-parameter refit

**NOT RUN.**

No new values exist for:

```text
K_p, K_v, k_a, K_R, K_omega, EI, Cb
```

No bounds were hit because no real optimization was launched. No Training objective or real-data continuity metric exists.

The old five Stage A parameter values remain historical window-based results. They are not promoted to the new model. A diagnostic synthetic excitation showed that the old high direct-model gains can become unstable under the revised attitude-coupled structure; this is not a real-data result, but it confirms that those values cannot be silently reused as if their interpretation were unchanged.

## G. Physical-only long-horizon validation

**NOT RUN.**

No free-run predictive metrics were computed for `fig8_003` or `osc_003`, because the physical fit did not pass the synthetic gate.

## H. Residual refit

**NOT RUN.**

The existing `90 -> 32 -> 32 -> 3`, 100-ms causal residual architecture remains preserved as a historical candidate. It was not retrained, renormalized, or attached to an unvalidated long-horizon physical model.

## I. Physics-plus-residual long-horizon validation

**NOT RUN.**

No comparison between Model P and Model PR exists for the new methodology.

## J. Old versus new interpretation

Milestone 3A results remain valid for their stated question: one-second open-loop prediction from frequently reinitialized measured states. They remain useful for lead-time error, command-response diagnostics, historical residual transfer, and debugging.

They do not establish tens-of-seconds free-run prediction. The new episode abstraction correctly removes internal measurement resets, but the constrained optimizer must first become computationally practical and pass inverse recovery before any long-horizon identification claim can be made.

## K. GUI and artifact status

The Dataset GUI now emphasizes:

- command propagation validity;
- independent UAV and cable observation validity;
- manual Include/Exclude loss masks;
- explicit Episode Break markers;
- Physical Episodes as the primary orange timeline object.

The historical overlapping windows are no longer the primary Dataset count.

The Identification GUI now shows:

- Coupled Physical ID: **BLOCKED AT SYNTHETIC RUNTIME GATE**;
- Causal UAV Residual: **LOCKED**;
- Provisional Long-Horizon Validation: **LOCKED**;
- Protected Test: **PENDING / NOT EVALUATED**;
- MPPI: **FUTURE**.

The historical protected-test button is disabled under the current workflow.

Stop artifacts are stored at:

```text
data/fit_results_long_horizon/20260828T125327_milestone3b_stopped/
```

No `UAV_CABLE_MODEL_FREEZE_LONG_HORIZON_PRETEST` was created.

## L. Protected test status

`fig8vertical_002` **WAS NOT USED FOR PREDICTIVE EVALUATION**.

It was assigned the `untouched_test` role and excluded from fitting, normalization, solver settings, acceptance decisions, and prediction metrics.

## M. Verification

| Check | Result |
|---|---|
| Python compilation | PASS |
| Eight structural tests | 8/8 PASS |
| Focused simulator/data/GUI suite | 40/40 PASS |
| Full repository suite | 76/76 PASS in 44.96 s |
| `git diff --check` | PASS (line-ending notices only for pre-existing tracked files) |
| GUI protected-test action | Disabled |
| Real physical fit | NOT RUN |
| Residual refit | NOT RUN |
| Protected predictive evaluation | NOT RUN |
| MPPI | NOT RUN |

## Required next decision

The scientific model and episode semantics do not need redesign based on this result. The blocker is the numerical optimization architecture/runtime of the current straightforward constrained solver.

Before repeating Milestone 3B, review a solver-only follow-up that preserves the exact scientific formulation but makes segment propagation and constrained optimization practical. Candidate engineering questions include batching/caching of segment evaluations, a reduced-memory adjoint/checkpoint strategy, a second-order constrained optimizer, and an explicitly measured pre-screening schedule for multistarts. That follow-up must rerun the same synthetic gate before real fitting.

## Final stop

Milestone 3B stopped at the required inverse-consistency gate. No conclusions are claimed about real seven-parameter identification, long-horizon prediction, residual value, or protected-test generalization.
