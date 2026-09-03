# PPO whip research handoff

Last updated: 2026-09-03

Branch: `twin-rewrite`
Repository: `particle_filter_cable_project`

## 1. Current decision

The current learned controller is **deterministic 10 Hz closed-loop PPO**. The
desktop Run & Replay page is deliberately pinned to:

`results/ppo/policies/PPO_WHIP_FORWARD_REVERSE_RELEASE_D50_V1/checkpoints/terminal.pt`

This checkpoint achieved 480/512 = **93.75%** success on a fixed bank of 512
distinct held-out, physically propagated initial UAV/cable states. It is the
preferred controller because it has the best verified success rate among the
recent compact-return experiments.

The newest return-focused candidate is retained at:

`results/ppo/policies/PPO_WHIP_DENSE_RETURN_RELEASE_100_V1/checkpoints/terminal.pt`

It achieved 474/512 = **92.58%**, with about 2.26 cm more return before impact
than D50. It is promising but is not the selected controller.

No policy in this repository is authorized for real flight.

## 2. Scientific system and task contract

- Frozen model: `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`.
- Production physics: UAV dynamics, frozen causal residual, and 12-node DDER.
- Physics step: 0.01 s.
- PPO query/control period: 0.10 s (10 Hz).
- Episode horizon: 7.0 s maximum.
- Termination: first successful strike or timeout; numerical failure is handled
  separately.
- PPO observation: normalized 83-D PolicyContext plus one remaining-time scalar
  (84 total).
- PPO action: target-aligned sagittal 3-D action: signed target-axis
  acceleration, signed vertical acceleration, and pitch rate.
- Commands are reanchored using the live simulated state, so this PPO is
  closed-loop. It is not a 49-D compiled open-loop action.
- Target: canonical fixed target for the retained policies.
- Physics parameters: nominal theta only.
- Validation policy: deterministic `tanh(mean)`.

Task success requires all of:

- cable endpoint enters within 0.050 m of the target;
- directed endpoint speed is at least 4.0 m/s;
- direction error is at most 30 degrees;
- c10 is the first cable marker entering the target region;
- finite numerical rollout.

UAV displacement and UAV-speed limits are currently diagnostics/costs, not hard
success gates. There is no 30--70-step strike-time gate.

## 3. Selected D50 reward

The selected controller uses one scalar reward:

- +20 times improvement in best world-frame tip progress;
- +60 times improvement in bounded joint strike quality;
- +100 on first task success;
- +50 times successful forward-load/return quality;
- +25 times successful backward-UAV/forward-cable release quality;
- -25 for a non-tip-first entry;
- -50 times smooth terminal displacement cost, charged only on success;
- -1 per pre-success second;
- -100 for numerical failure.

Directed-speed shaping uses cable-tip velocity relative to the physical
attachment and saturates at 4.0 m/s. The displacement integral, acceleration
effort, body-rate effort, and action-smoothness weights are zero.

The return quality requires a real positive forward excursion followed by
target-axis return. The release quality requires the UAV to move backward while
the cable tip moves forward relative to its attachment.

## 4. Selected D50 result

Run source:
`data/policy_training/whip_ppo_forward_reverse_release_v1/2026-09-02T174750.625177Z`

Tracked policy folder:
`results/ppo/policies/PPO_WHIP_FORWARD_REVERSE_RELEASE_D50_V1`

Deterministic 512-state validation:

| Metric | Result |
|---|---:|
| Task success | 480/512 = 93.75% |
| Median tip error | 30.44 mm |
| Median directed speed | 5.10 m/s |
| Median direction error | 21.29 deg |
| Median strike time | 0.88 s |
| Mean maximum UAV displacement | 0.916 m |
| Mean terminal UAV displacement | 0.829 m |
| Mean peak forward displacement | 0.625 m |
| Mean forward displacement at impact | 0.529 m |
| Mean return before impact | 0.0966 m |
| Mean UAV forward velocity at impact | -1.445 m/s |
| Mean relative tip forward speed at impact | 6.580 m/s |
| Numerical failures | 0 |

The mechanism audit confirms that successful trajectories generally load
forward, reverse, and hit while the UAV is already moving backward. The policy
does not return all the way to its initial position before the strike.

## 5. Return/displacement studies

### D60 terminal displacement

Increasing terminal displacement weight from 50 to 60 was trained through
600,064 cumulative episodes.

- Best guarded checkpoint: 461/512 = 90.04%.
- Mean maximum displacement: 0.896 m.
- Mean terminal displacement: 0.762 m.
- Mean return: 0.117 m.
- Latest scheduled validation: 88.67% with 0.126 m return.

Conclusion: generic displacement pressure increased return but mainly caused
extra direction-gate failures. The small maximum-excursion reduction did not
justify the success loss.

### Dense return/release shaping

The implementation now supports a bounded dense potential that becomes positive
only when all of these occur together:

1. genuine forward loading;
2. return from the peak along the target axis;
3. backward UAV velocity;
4. forward cable-tip velocity relative to the attachment;
5. nonzero instantaneous strike quality.

The potential is paid through improvement in the episode-best value, preventing
repeated reward accumulation by cycling. A configuration can also require the
terminal release bonus to use release quality at the actual impact instead of an
earlier episode-best value.

The 50-weight pilot reached 91.99% with 0.103 m mean return. Continuing with
dense weight 100 for 501,760 episodes produced:

| Metric | D50 selected | Dense-return final |
|---|---:|---:|
| Validation success | 93.75% | 92.58% |
| Mean maximum displacement | 0.916 m | 0.915 m |
| Mean terminal displacement | 0.829 m | 0.806 m |
| Mean return before impact | 0.0966 m | 0.119 m |
| Median directed speed | 5.10 m/s | 5.20 m/s |
| Median direction error | 21.29 deg | 21.72 deg |

The final dense-return checkpoint is the best current success/return trade-off,
but D50 remains selected because its success rate is higher.

### Failed progress-reference experiments

- A fresh 50/50 blend of world-tip and attachment-compensated progress peaked at
  45.90% validation success early, then collapsed. It did not produce meaningful
  return.
- Pure attachment-compensated progress from random weights found a runaway
  translation solution. The latest validation was 0/512 and mean UAV
  displacement grew to roughly 49.5 m.
- These runs are stopped and must not be resumed as current baselines.

Compact histories and plots are in `results/ppo/reward_studies/`.

## 6. Validation interpretation

The 512 validation episodes do **not** repeat the same initial state. The manifest
is a permutation of 512 distinct held-out state-bank rows. Initial UAV velocity,
orientation, angular velocity, cable shape, distributed cable velocity, and tip
motion vary. The target and nominal theta remain fixed.

For the final dense-return validation, most failures were associated with the
30-degree direction gate rather than speed: 56 of 58 failures did not record a
passing direction result, while median successful directed speed remained about
4.99--5.20 m/s. Gate-failure counts overlap when a trajectory never records a
valid tip-first entry.

## 7. PPO, SAC, CEM, and diffusion status

- PPO: current learned-controller path. Closed-loop at 10 Hz.
- SAC: retained negative baseline under `results/sac`; stopped after about 1.2M
  episodes with 0.0088% cumulative success.
- CEM: retained offline reference under `results/cem`; 98.05% first-seed and
  98.44% with up to three seeds, but about 34.6 s median planning time.
- Diffusion/amortized CEM: investigated extensively. A DDIM terminal scheduling
  bug was repaired, but the frozen generator remained candidate-support/state-
  conditioning limited. It is not the current controller.
- PPO+sim open-loop compiler: exact nominal replay works, but batch-one
  compilation is about 12.2 s and feedback is materially more robust under model
  mismatch/disturbance. Closed-loop PPO was therefore selected.

## 8. Implementation map

- PPO runner and configuration validation: `run_simple_ppo.py`.
- Shared environment and reward: `learning/sequential_sac_env.py`.
- PPO implementation: `learning/simple_ppo.py`.
- State-bank validation and plots: `learning/ppo_validation.py`.
- Mechanism audit: `run_ppo_whip_mechanism_audit.py`.
- Desktop training UI: `simulator/gui/training_page.py`.
- Selected-policy replay UI: `simulator/gui/replay_page.py`.
- Production replay command: `run_ppo_simulation.py`.
- Current experiment configurations: `config/learning/whip_ppo_*.json`.

The training UI exposes progress/speed references, return/release terms, dense
return-release strength, release-at-strike timing, terminal displacement,
training parameters, validation cadence, rolling-window size, configuration
save/load, and manual validation. Training and validation curves are separated
into tabs. The Run & Replay page is pinned to the tracked D50 policy rather than
the newest timestamped run.

## 9. Reproduction commands

For a fresh clone or RTX 5090 workstation, complete
[`PORTABLE_WORKSTATION_SETUP.md`](PORTABLE_WORKSTATION_SETUP.md) first. The
committed `.run/` configurations are the preferred PyCharm entry points.

From PowerShell in the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run_simulator.py
```

Preflight a configuration:

```powershell
.\.venv\Scripts\python.exe verify_workstation.py
.\.venv\Scripts\python.exe run_simple_ppo.py --preflight --config config\learning\whip_ppo_portable_continuation_v1.json
```

Start a new configured run:

```powershell
.\.venv\Scripts\python.exe run_simple_ppo.py --train --config config\learning\whip_ppo_portable_continuation_v1.json
```

Do not resume completed failed progress-reference runs. Timestamped raw runs live
under `data/policy_training` and are intentionally excluded from Git. The
selected checkpoints, compact logs, plots, and manifests are tracked under
`results/ppo` for portability.

The immutable production model freeze and the small fit summaries required by
the simulator are also tracked explicitly. Historical experiment configs are
retained as provenance and may reference archived timestamped parent runs; use
`whip_ppo_portable_continuation_v1.json` for new work on another computer.

## 10. Recommended next step

Avoid another blind reward sweep. First run one controlled, read-only comparison
of the tracked D50 terminal checkpoint and dense-return terminal checkpoint on
the identical 512-state bank, including low/medium/high state-distance strata
and the full mechanism audit. Decide whether the 1.17-point success loss is worth
the 2.26 cm additional return.

If neither trajectory is sufficiently compact, the next training change should
target **target-axis position at impact** or a staged maneuver representation,
not a larger isotropic displacement penalty. D60 demonstrated that generic
displacement pressure primarily damages strike direction.

Only after selecting one canonical policy should work proceed to held-out target
generalization. Theta conditioning, Real-to-Sim adaptation, and hardware remain
future stages.

## 11. Safety and untouched scopes

- Protected `fig8vertical_002`: **NOT EVALUATED**.
- Real hardware: **NOT EXECUTED**.
- Model fitting during these PPO reward studies: **NONE**.
- CEM used for PPO initialization/demonstrations: **NO**.
- SAC used in PPO training: **NO**.
