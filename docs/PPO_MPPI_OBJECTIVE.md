# PPO with the selected MPPI objective

10 September 2026. User-authorized separate PPO simulation experiment.

## What changed

PPO previously read the legacy incremental `reward` fields. The selected MPPI
job actually uses `PreferredFoldCapture` in `planning/whip_objective.py` and its
`trajectory_objective` weights. Merely copying `reward` would not match the task.

The new versioned PPO mode calls that same whole-trajectory scorer in training,
deterministic evaluation and policy replay. It freezes the exact shape reference
and checks its SHA-256. The prior remains a user-preferred shape history from
simulation, not a validated physical wave detector or measured training data.

The selected MPPI command/forecast and flight selection are untouched.
There is no new model fitting, real flight, M1 or adaptation claim.

## Current trial

Status read during the subsequent command-chain audit: completed at 491,520
attempts by `reward_plateau`, best deterministic fixed-scenario score 692.8127
with modeled contact. This is not a convergence proof or real-flight result.
The final policy was not independently rehearsed by that audit; the separately
frozen replay checks below retain their original checkpoint identities.

- Run: `runs/ppo_pva/20260910-115619-908807`; consult `status.json`, `history.json`
  and `checkpoints/best_evaluation.json` for current results.
- Fresh PPO policy, seed 655; no MPPI action imitation or historical policy.
- Exact same development M0 as MPPI `20260910-022818-648386`: inherited drone fit,
  scalar cable damping 0.4/s, curvature regularization 2e-5, cable NN disabled.
  Incomplete-fit/development provenance is retained. The UI's explicit development
  review is bound to the source model checksum, not transferable to other models.
- Start `[0,0,1.255]` m; target `[1.25,0,1.0]` m; 5 cm sphere. Fixed start/target.
- Whole maneuver 1.5 s, 45 jerk actions at 30 Hz, 150 Hz physics. PPO has no MPPI
  lookahead or candidate sampling horizon. Exact selected MPPI limits remain.
- Success is feasible tip entry (`tip_contact_v1`). Wave/reversal diagnostics do
  not veto contact. The soft reward still prefers fold, reversal and outward cast.
- 2,048 parallel CUDA environments; fused ticks and specialized cable solvers.
  Policy/optimizer FP32, physical model FP64. Hidden width 256, learning rate
  3e-4, four PPO epochs, minibatches 8,192, entropy coefficient .005.
- Gamma 1 and GAE lambda 1 preserve full episodic credit, reward scale .01.
  Evaluation every four updates. With fixed launch, one deterministic evaluation
  avoids 128 identical rollouts. Its 0/100% indicator is one scenario, not a
  statistically independent robustness estimate.
- Best checkpoint selected by the shared objective, not hit-rate-first ranking.
  Minimum 98,304 attempts, plateau patience 49,152 attempts, relative improvement
  .005, review ceiling 500,000. Check actual stop reason; a ceiling is not convergence.

## Reward and credit assignment

Feasible trajectory scores match MPPI exactly: contact 450, preferred fold 600,
cast 200, miss 300, approach 15, lateral 30, jerk .02, exit speed .5, exit climb 5,
exit forward acceleration 1. The copied MPPI hit bonus is zero. Legacy incremental
reward is retained only for compatibility/diagnostics; it is not optimized here.

MPPI gives infeasible trajectories negative infinity. PPO must use finite returns,
so every failed trajectory receives -10,000. This is a finite surrogate for MPPI
rejection, not an assertion that the two stochastic optimizers solve the same
constrained problem. Feasibility is reported separately and final rehearsal still
rejects an infeasible complete command. The penalty is fixed within the run.

For training only, define a causal potential
`Phi_t = -300 * closest_tip_distance_so_far / cable_length`.
Use `Phi_(t+1) - Phi_t` and add the shared score at the terminal transition, with
terminal potential set to zero. At gamma 1 the sum is `score - Phi_initial`:
only an initial-state constant differs, including early hits and failures. Logged
returns and checkpoint selection use the original shared score. Masks exclude
all transitions after termination. Partial rollouts cannot be assigned this score.
This redistributes credit without adding a new terminal target preference.

The whole-trajectory shape objective is history dependent. It is computed after
collection, like an episodic return; future state is not fed to the policy. The
existing feedforward policy receives the physical observation, time, command
history and phase credits, not the entire shape history. No claim of a sufficient
Markov state for that history-dependent reward is made.

## Verification and UI

Audit: `runs/audits/ppo-mppi-objective-20260910`.

- Windows / RTX 4080 / PyTorch 2.11.0+cu128: replayed the selected MPPI commands
  through independent MPPI and PPO paths. Both scored **695.153910576967**;
  every reward component, command packet, termination, success and minimum
  distance agreed (1e-8 absolute tolerance). This verifies plumbing, not that
  PPO has learned those commands.
- A 128-environment stochastic rollout and optimizer update passed finite-value,
  early-termination mask and telescoping-return checks.
- Focused suite: 59 passed, one skipped (missing historical fixture).
- GPU usage observed at 96% during the 2,048-environment trial. This is a sampled
  utilization reading, not a claim of theoretical maximum throughput.
- A running checkpoint was frozen and replayed using its original source/model
  plus the checksummed shape reference; training continued. That first software
  replay was a miss and remains saved in the audit.
- Native Qt dashboard verified. Open **PPO → Training progress** for attempts,
  policy tests, distance, infeasibility, curves and logs. **Rehearse policy** takes
  a snapshot without stopping training; **Refresh policy rehearsal** loads newer
  learning. Keep “Include recovery / export” off when inspecting an unfinished
  policy. These previews do not replace the selected MPPI flight forecast.

Training results must distinguish stochastic sampled hits from the deterministic
policy, target contact from preferred whip shape, and modeled feasibility from
real flight validation. Consult the live run before reporting learning success.

## First independently replayed learned result

The checkpoint at 98,304 attempts scored 401.3043697811321, with modeled tip entry
at about 1.255 s and backward release. Its trace replay reproduced the deterministic
evaluation score exactly. Complete recovery passed the existing predicted command,
pose/cable parity and envelope checks. This is a learned PPO trajectory, not the
MPPI parity-test command. Its preferred-fold component (102.88) is below the
selected MPPI's 196.21. Contact alone does not establish the desired wave quality.

Saved independently in
`runs/rehearsals_pva/20260910-120253-429897-ppo-matched-objective` (complete recovery)
and the sibling `-preview` folder. The original MPPI flight remains selected.
Subsequent checkpoint reviews are indexed by
`runs/audits/ppo-mppi-objective-20260910/latest_policy_review.json`; training can
continue beyond these frozen reviews. No claim of real PPO flight performance.

At 139,264 attempts, the independently replayed checkpoint scored **560.3134639757728**
and hit the same target; complete recovery again passed. Its fold component was
91.38 versus MPPI's 196.21, so its higher total mainly reflects contact/cast and
does not demonstrate equally good preferred-wave shape. Complete rehearsal:
`runs/rehearsals_pva/20260910-120514-716759-ppo-matched-objective`.

Source-only exports retain the reward/reference checksum but do not bundle its
research asset or carry over the workstation's development-model authorization.
Supply the separately reviewed `config/pva/wave_reference.npz` when reproducing
this experiment outside the original frozen job.
