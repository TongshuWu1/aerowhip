# Amortized CEM One-Shot Policy Pilot Report

Date: 2026-08-30  
Artifact: `data/policy_training/amortized_cem_nominal_pilot_v1/2026-08-30T063828.195418Z`

## 1. Executive result

The pure-SAC branch has been replaced in the active pipeline by **amortized
trajectory optimization**:

\[
U^*(x_0,g,\theta)=\operatorname{CEM}(x_0,g,\theta),\qquad
\pi_\psi(x_0,g,\theta)\approx U^*(x_0,g,\theta).
\]

CEM runs offline to generate complete-maneuver labels. A feed-forward actor is
trained by supervised regression. At deployment there is one actor query and
no CEM, critic, replay buffer, entropy objective, or feedback query.

The first modest pilot produced two distinct conclusions:

1. **Canonical amortization passed.** The actor's deterministic canonical
   action passed every scientific gate with a 1.11475-s active maneuver and a
   valid strike at 1.12000 s. This is a direct demonstration that the new
   maneuver/evaluation-time separation works: the cable struck after the
   active maneuver ended while the vehicle command was already in terminal
   hold.
2. **Interpolation did not yet pass.** Among nine CEM-solvable held-out test
   contexts, actor success was 0%. The pilot therefore does not establish a
   general policy. The verified training set contained only 14 labels, and
   nearby CEM labels were substantially less batch-one robust than their
   population results.

Classification:

```text
CANONICAL_AMORTIZATION: PASS
HELD_OUT_INTERPOLATION: FAIL
GENERAL_POLICY: NOT YET ESTABLISHED
```

This is still a cleaner scientific outcome than the SAC branch: the active
software now tests the intended mapping directly, and the remaining question
is teacher-data coverage/consistency rather than critic or exploration design.

## 2. Code cleanup and zombie-code audit

The cleanup was conservative because the repository contains scientific
artifacts whose source-hash manifests refer to the completed SAC experiments.
Deleting those sources would reduce reproducibility.

### 2.1 Removed from the active code path

`learning/__init__.py` previously advertised SAC as if it were the current
learning API. The following obsolete public exports were removed:

- `OneShotActor`
- `OneShotCritic`
- `PolicySample`
- `ReplayBatch`
- `TerminalReplayBuffer`
- `TerminalSacAgent`
- `radial_squash`
- `squash_raw_action`

The active package now exports the supervised components:

- `AmortizedTrajectoryActor`
- `SupervisedTrainingResult`
- `train_amortized_actor`

This prevents new application code from accidentally selecting the abandoned
SAC stack through the package's normal import surface.

### 2.2 Active modules

The current learning/planning path uses:

| Layer | Active source |
|---|---|
| policy context | `learning/policy_context.py` |
| action codec | `learning/policy_action.py` |
| fixed context normalization | `learning/normalization.py` |
| one-shot simulator boundary | `learning/one_shot_env.py` |
| physically valid state banks | `learning/state_bank.py` |
| context specifications | `learning/context_sampling.py` |
| supervised actor | `learning/amortized_policy.py` |
| offline CEM teacher | `planning/teacher_cem.py` |
| variable-duration commands/evaluation | `planning/variable_duration.py` |
| experiment entry point | `run_amortized_cem_pilot.py` |

### 2.3 Historical code retained deliberately

The following modules are not imported by the active package, production GUI,
teacher runner, supervised trainer, or deployment evaluator:

```text
learning/sac.py
learning/replay.py
learning/training.py
learning/canonical_pilot.py
learning/pilot_training.py
learning/spectral_sac.py
learning/spectral_pilot.py
learning/constrained_sac.py
learning/safety_quarantine.py
learning/reward_audit.py
learning/rollout_diagnostics.py
```

They are retained as **retired reproducibility code** for Milestones 5B through
5B.6, not as current implementation. The historical `run_milestone5b*.py`
runners likewise remain reproducibility entry points only. They should receive
no new features.

The production GUI does not import the learning package. It only loads saved
planning/replay artifacts. No SAC object is constructed by the current runner.

The machine-readable audit is at
`reports/amortized_cem_zombie_code_audit.json`; the active/retired boundary is
documented in `learning/README.md`. The root README was also corrected to name
the actual frozen model, four-page GUI, and current amortized-CEM method.

Result:

```text
ACTIVE_SURFACE_CLEANED
HISTORICAL_BRANCH_RETAINED_FOR_REPRODUCIBILITY
```

## 3. Why the method was reset

The previous SAC experiments asked random or locally stochastic 48/49-D
actions to discover a narrow, coordinated whip manifold using one terminal
return. Reward conditioning, critic stability, temporal spectral exploration,
fixed duration, safety quarantine, and constrained critics were each tested.
None produced a deterministic generalizable whip, and no stochastic scientific
success was found.

CEM already demonstrated that the same frozen simulator, action representation,
and scientific task possess a feasible solution. The neural network is
therefore assigned the easier and more appropriate problem: interpolate among
verified optimized solutions. This amortizes offline optimization cost without
turning CEM into the online controller.

The research object remains the one originally intended:

\[
(x_0,g,\theta)\xrightarrow{\text{one network query}}U.
\]

## 4. Frozen production model

The pilot used exactly:

```text
MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI
```

Unchanged components:

- fitted UAV Physics gains `K_p`, `K_v`, `k_a`, `K_R`, `K_omega`;
- frozen causal UAV residual, normalization, and residual FIFO;
- cable `EI`, `Cb`, measured geometry, masses, and rigid attachment;
- 12-node isotropic DDER;
- CUDA float32;
- PCG32 damping solve;
- three DDER substeps;
- four position projections;
- production fused forward path;
- fixed UAV/residual evaluation shape of 2048.

Physics context was nominal only. No parameter fitting, randomization, residual
change, controller change, or hardware execution occurred.

## 5. Detailed current pipeline

### 5.1 Offline data flow

```text
frozen simulator + physically valid state banks
    |
    +--> choose split-disjoint initial state x0
    |
    +--> choose root/yaw-local target and desired direction g
    |
    +--> build structured PolicyContext(x0, g, theta_nominal)
    |
    +--> warm-start local variable-duration CEM from one consistent
    |    canonical whip family
    |
    +--> population rollout in frozen production simulator
    |
    +--> scientific hard-gate selection
    |
    +--> authoritative batch-one replay of selected action
    |
    +--> retain label only if batch-one replay also passes
    |
    +--> dataset row (83-D context, 49-D optimized maneuver)
    |
    +--> supervised actor training
    |
    +--> frozen-simulator validation/test of actor predictions
```

### 5.2 Policy context: 83 dimensions

The policy context is constructed exactly once at the query boundary in a
root-centered, yaw-aligned, gravity-preserving frame.

The origin is the current physical cable connector/root. Initial UAV yaw is
removed while local +Z remains aligned with world gravity. World translation
and global yaw therefore do not need to be learned.

Tensor contents:

| Feature | Shape | Dimensions |
|---|---:|---:|
| UAV linear velocity, local | 3 | 3 |
| UAV orientation, local xyzw quaternion | 4 | 4 |
| UAV angular velocity, local | 3 | 3 |
| c1...c10 positions relative to root | 10 x 3 | 30 |
| c1...c10 velocities | 10 x 3 | 30 |
| target position relative to root | 3 | 3 |
| desired strike direction | 3 | 3 |
| `K_p`, `K_v`, `k_a`, `K_R`, `K_omega`, log(EI), log(Cb) | 7 | 7 |
| **Total** |  | **83** |

Quaternion convention remains production `xyzw`; sign is canonicalized so
`q` and `-q` do not form two representations. The previously fitted fixed
normalizer, estimated from 100,000 nominal training contexts, is reused. It is
not refit from this small pilot.

### 5.3 Action: one complete 49-D maneuver

The actor output is:

```text
16 acceleration knots x 3 axes = 48
active maneuver duration T      = 1
total                            = 49
```

The output lives in `[-1,1]^49`. Acceleration vectors are decoded with the
existing vector-norm projection so every knot satisfies
`||a_k|| <= 20 m/s^2`. The duration coordinate maps deterministically to
`[0.45,1.20] s`.

Knots are expressed in the query-time local frame and rotated to world before
command construction. Piecewise-linear acceleration is integrated once to
produce mutually consistent command acceleration, velocity, and position.
The actor never predicts these three quantities independently. Yaw is held at
the query-boundary convention and commanded angular velocity is zero.

### 5.4 Separate maneuver and evaluation time

The current contract distinguishes:

```text
T_maneuver   in [0.45, 1.20] s  (optimized and labeled)
T_evaluation = 1.50 s           (fixed numerical observation window)
```

For `t <= T_maneuver`, the integrated acceleration trajectory is commanded.
For `t > T_maneuver`, the FullState command is a terminal hold:

```text
p_cmd = final maneuver command position
v_cmd = 0
a_cmd = 0
yaw_cmd = query-boundary yaw
omega_cmd = 0
```

Cable and UAV physics continue to propagate through `T_evaluation`, so a
delayed distal lash may strike after the active maneuver ends. The 1.20-s
maneuver bound is not treated as a scientific hit deadline.

This distinction was not cosmetic: 9 of 32 verified teacher labels struck
after their active maneuver duration. The final canonical learned action also
ended at 1.11475 s and struck at 1.12000 s.

### 5.5 CEM teacher

Each context is solved independently and offline using:

| Setting | Value |
|---|---:|
| population | 2048 |
| elite fraction | 5% |
| maximum iterations | 15 |
| post-first-success iterations | 3 total including success iteration |
| acceleration initial standard deviation | 1.0 m/s² |
| duration initial standard deviation | 0.025 s |
| acceleration standard-deviation floor | 0.15 m/s² |
| duration standard-deviation floor | 0.005 s |
| old/elite distribution weights | 0.30 / 0.70 |
| covariance | full 49 x 49 |
| population numerical shape | exactly 2048 |
| evaluation time | 1.50 s |

Every solve begins at the same saved tuned-CEM seed-46 maneuver family. No
validation/test solution is warm-started from a training solution, and no
solution crosses split boundaries. This deliberately reduces label
multimodality: neighboring contexts are encouraged to remain in the same
forward/reversal/lash family instead of producing arbitrary incompatible CEM
modes.

Teacher ranking is global across the population and feasibility-first. A
label must pass every unchanged hard scientific gate. Population success alone
is insufficient: the selected physical decision is encoded through the exact
policy codec and replayed at batch one. Only batch-one successes enter the
dataset.

### 5.6 Split construction

The pilot reused existing nominal, physically propagated state banks; marker
positions were never independently perturbed.

Designed contexts:

| Split | Distinct state rows | Targets per state | Extra canonical | Total |
|---|---:|---:|---:|---:|
| train | 16 nearest training-bank rows | 2 | 1 | 33 |
| validation | 8 nearest validation-bank rows | 2 | 0 | 16 |
| test | 8 different validation-bank rows | 2 | 0 | 16 |

Training, validation, and test state rows are disjoint. Target offsets are
deterministic, antithetic, and confined to +/- `[0.035,0.035,0.020]` m around
the canonical local target. Desired direction is recomputed as the horizontal
unit vector from the root toward each target.

### 5.7 Supervised actor

Architecture:

```text
normalized context [83]
    -> Linear 256 -> SiLU
    -> Linear 256 -> SiLU
    -> Linear 256 -> SiLU
    -> Linear 49  -> tanh
    -> complete normalized maneuver [49]
```

Training is ordinary AdamW supervised regression:

```text
loss = MSE(normalized acceleration coordinates)
     + MSE(normalized duration coordinate)
```

Settings:

- learning rate: `3e-4`;
- weight decay: `1e-5`;
- minibatch: 32;
- maximum epochs: 5000;
- validation patience: 500 epochs;
- gradient-norm limit: 10;
- checkpoint selection: minimum validation imitation loss.

There is no simulator gradient. There is no task reward inside supervised
training. The actor learns only CEM labels. Actual policy quality is decided
afterward by the frozen production simulator and hard gates, not by action MSE.

### 5.8 Deployment/evaluation path

```text
one measured/simulated initial state x0
    + one target/direction g
    + current model context theta
        -> root/yaw-local PolicyContext [83]
        -> fixed normalization
        -> one feed-forward actor call
        -> normalized complete action [49]
        -> deterministic physical decoder
        -> complete FullState command generated before execution
        -> open-loop frozen production rollout
        -> scientific hard-gate evaluation
```

No policy function is accepted by the physics loop. There is no place in this
API for `policy(state_t)`, feedback correction, replanning, or receding horizon.
CEM is absent from this path.

## 6. Reward and scientific evaluator

The offline CEM teacher uses the same accepted optimizer/reference reward:

```text
legacy_run_online_strike_margin_tuned_v4
```

This is the reward previously tuned for the successful CEM maneuver. The
supervised actor does not optimize that reward directly; it regresses teacher
actions. Final acceptance is always the unchanged hard-gate evaluator:

- tip-target distance <= 0.050 m;
- directed tip speed >= 4.0 m/s;
- impact direction error <= 30 deg;
- c10 enters first;
- UAV displacement <= 0.50 m;
- UAV speed <= 3.0 m/s;
- command acceleration <= 20.0 m/s²;
- finite rollout.

The experimental RL rewards `rl_whip_reward_v1/v2` remain historical and are
not used by this pilot.

## 7. Teacher-generation result

All 65 CEM population solves found a scientific success. However, only 32
selected actions remained successful under authoritative batch-one replay.

| Split | Designed | Population CEM success | Verified batch-one labels |
|---|---:|---:|---:|
| train | 33 | 33 | 14 |
| validation | 16 | 16 | 9 |
| test | 16 | 16 | 9 |
| **total** | **65** | **65** | **32** |

Additional teacher statistics:

- population rollouts: 428,032;
- teacher optimization runtime: 360.89 s;
- 51 contexts stopped after 3 iterations;
- 14 contexts stopped after 4 iterations;
- verified teacher duration range: 1.08075 to 1.15052 s;
- verified teacher median duration: 1.11231 s;
- verified delayed strikes (`t_hit > T_maneuver`): 9 of 32.

The 49.23% batch-one verification rate is the most important teacher-data
finding. Population success was easy near the canonical mode, but many labels
were numerically fragile across population vs deployment batch shape. They
were excluded rather than treated as expert data. A larger dataset campaign
should first improve this teacher/replay consistency or optimize directly
under the authoritative deployment replay contract.

## 8. Supervised fit

Verified labels used:

```text
train      14
validation  9
test        9 (never used for gradients or checkpoint selection)
```

Best checkpoint:

- epoch: 88;
- epochs executed before patience stop: 588;
- best validation loss: 0.00436466;
- training normalized action RMSE: 0.04212;
- validation normalized action RMSE: 0.04435;
- validation normalized acceleration RMSE: 0.04425;
- validation normalized duration RMSE: 0.04906.

The low imitation error did not translate into held-out task success. This
confirms that action MSE is only a secondary diagnostic for an aggressive,
narrow-manifold maneuver.

## 9. Policy evaluation

Evaluation was deterministic and batch one. Every row was a context for which
an independently optimized teacher label had already passed batch-one replay.

| Split | Count | Hard success | Feasible | Median tip distance | Median directed speed | Median direction error |
|---|---:|---:|---:|---:|---:|---:|
| train | 14 | 21.43% | 57.14% | 98.23 mm | 4.193 m/s | 25.03 deg |
| validation | 9 | 0.00% | 55.56% | 102.10 mm | 4.100 m/s | 25.70 deg |
| test | 9 | 0.00% | 66.67% | 241.45 mm | 4.173 m/s | 28.31 deg |

The actor often preserved useful speed and approximate direction but missed
the small spatial strike manifold and sometimes violated the tight UAV
displacement bound. This is consistent with averaging a small set of slightly
different CEM actions: small action errors can create large phase errors at
the cable tip.

### 9.1 Canonical authoritative result

The canonical actor prediction was replayed once through complete production
physics for 1.50 s.

| Hard gate | Limit | Result | Pass |
|---|---:|---:|---:|
| c10 target distance | <= 50 mm | 48.910 mm | YES |
| directed tip speed | >= 4.0 m/s | 4.1455 m/s | YES |
| impact direction error | <= 30 deg | 18.590 deg | YES |
| tip first | c10 | c10 | YES |
| UAV displacement | <= 0.50 m | 0.49619 m | YES |
| UAV speed | <= 3.0 m/s | 2.26005 m/s | YES |
| command acceleration | <= 20.0 m/s² | 16.0321 m/s² | YES |
| finite | required | finite | YES |

Timing:

```text
active maneuver duration = 1.1147466 s
valid hit time           = 1.1200000 s
evaluation window        = 1.5000000 s
hit after maneuver       = YES
```

This canonical pass is real but has narrow margins in target distance and UAV
displacement. It must not be presented as a robust or hardware-ready policy.

## 10. CEM reference

The original saved canonical CEM seed-46 result remains the teacher-family
anchor and feasibility reference:

```text
tip error       ~1.756 mm
directed speed  ~4.599 m/s
direction error ~19.615 deg
duration        ~1.1175 s
scientific task PASS
```

The actor was not evaluated by matching those exact decimals. Its own complete
production replay determines success.

## 11. Verification

The complete repository regression suite passed after the pipeline and active
surface cleanup:

```text
103 passed in 39.44 s
```

Focused new checks cover:

- actor input/output shape and bounded normalized output;
- separation of acceleration and duration imitation loss;
- post-maneuver terminal FullState hold;
- distinct 1.20-s maneuver bound and 1.50-s evaluation window;
- explicit absence of SAC and online CEM in the active configuration.

Existing tests continued to cover the frozen production model, variable CEM,
one-shot context/action contracts, SAC historical reproducibility, GUI replay,
and numerical production backends.

## 12. Artifacts

The run directory contains:

```text
config.json
context_schema.json
context_normalizer.json
context_split_manifest.json
teacher_attempts.json
verified_teacher_labels.json
teacher_dataset.npz
teacher_checkpoints/
best_actor.pt
supervised_training_history.json
supervised_training_summary.json
policy_evaluation_rows.json
evaluation_summary.json
canonical_predicted_normalized_action.npy
canonical_fullstate_command.csv
canonical_final_replay.npz
canonical_final_metrics.json
amortized_cem_actor_canonical_final_replay.mp4
video_metadata.json
source_hash_manifest.json
run_summary.json
```

Total end-to-end runtime was 550.56 s. No CEM optimization occurs when loading
or querying `best_actor.pt`.

## 13. Scientific interpretation

The reset is methodologically sound, and the software now represents the
intended deployment object directly. The canonical pass shows that a small
feed-forward actor can reproduce a complete aggressive maneuver sufficiently
well for one known context. The held-out failure shows that this first pilot
is not enough to establish interpolation.

The evidence points to two immediate data-method issues, not a reason to
restore SAC complexity:

1. **Teacher robustness:** only 32 of 65 population-success actions were
   batch-one successes. Teacher labels should be optimized and selected under
   the exact authoritative replay shape, or otherwise made numerically
   consistent before scaling the dataset.
2. **Dataset size and mode smoothness:** 14 training labels are far too few for
   a trustworthy 83-D to 49-D map, even in a narrow domain. More verified labels
   should be generated along a deliberately consistent maneuver branch. If
   nearby labels still show discontinuous modes, a mode-aware target
   representation can be considered later; it is not justified yet.

The next experiment should remain simple: improve teacher label consistency,
generate a materially larger split-disjoint nominal dataset, train the same
actor, and judge it exclusively by held-out simulator success. Do not add a
critic, entropy term, replay buffer, or new reward.

No next experiment was started automatically.

## 14. Final summary

```text
Model:
    MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

Method:
    AMORTIZED CEM TRAJECTORY OPTIMIZATION

Offline optimizer:
    Variable-Duration CEM

Online optimizer:
    NONE

Policy:
    feed-forward 83 -> 256 -> 256 -> 256 -> 49

Policy query count:
    1

Execution:
    OPEN LOOP

Context:
    UAV + c1...c10 position/velocity + goal + nominal theta

Action:
    16 x 3 acceleration knots + active maneuver duration

Maneuver duration range:
    [0.45, 1.20] s

Evaluation time:
    1.50 s

Post-maneuver behavior:
    TERMINAL FULLSTATE HOLD; CABLE PROPAGATION CONTINUES

Teacher reward:
    legacy_run_online_strike_margin_tuned_v4

Supervised loss:
    normalized acceleration MSE + normalized duration MSE

Designed contexts:
    train 33 / validation 16 / test 16

Verified labels:
    train 14 / validation 9 / test 9

Teacher population success:
    65 / 65

Teacher batch-one verified success:
    32 / 65

Canonical learned policy:
    PASS
    tip distance = 48.910 mm
    directed speed = 4.1455 m/s
    direction error = 18.590 deg
    maneuver duration = 1.11475 s
    hit time = 1.12000 s

Held-out test:
    success = 0.00%
    feasible = 66.67%

Canonical amortization:
    PASS

Held-out interpolation:
    FAIL

General one-shot policy:
    NOT YET ESTABLISHED

SAC:
    NOT USED

CEM at deployment:
    NO

Physics conditioning learned:
    NO — NOMINAL ONLY

Production model modified:
    NO

Protected test:
    NOT EVALUATED

Real hardware:
    NOT EXECUTED
```
