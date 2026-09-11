# M2 PPO with sustained exploratory actions

11 September 2026. User authorized the exploration improvement after reviewing
why the first M2 PPO learned forward contact without the preferred backward
release. Job: `runs/ppo_pva/20260911-142454-400090`. Audit and supervisor:
`runs/audits/M2-ppo-persistent-20260911`. Read their status files before acting.
One fresh run is authorized and launched; no automatic restart or retuning.

## Frozen experiment

The exact M2-frozen-refit-v1 model, both neural residuals, initial state, target,
1.5 s episode, physics, limits, preferred-fold shape reference, reward weights,
potential shaping and contact-first checkpoint selection match the previous M2
PPO. Start [0,0,1.255] m; target [1.25,0,1.00] m; target radius 5 cm.
The impact weight remains 1600, with speed scale 4 m/s. No reward changes.

The new versioned `fixed_jerk_hold_v1` policy timing holds each sampled XYZ jerk
for three control steps (0.1 s). This gives at most 15 policy decisions per
1.5 s attempt, instead of 45. PVA integration and packet output remain 30 Hz;
physics and contact/feasibility checks remain 150 Hz. Early contact or failure
can terminate inside a block. Rehearsal uses the same policy timing. Explicit
command sequences bypass holding, so replay never applies the hold twice.

This changes the policy's available action sequences as well as its exploration;
it is not equivalent to merely running the old policy for longer. The purpose
is to reduce rapidly cancelling exploratory actions and enable sustained motion.
Three steps is an engineering choice, not a proven optimal whip timescale.
No cast/reversal direction is prescribed. All three action axes remain learned.

Fresh random actor, critic and optimizer; seed 655. No old PPO weights or MPPI
demonstration commands initialize training. The original shape-preference reward
already includes an archived motion reference, exactly as in the matched baseline;
from-scratch here does not mean learning without any task prior.

Training uses 2,048 CUDA environments, the same 256-wide networks, initial log
standard deviation -0.8, learning rate 3e-4, four maximum PPO epochs, minibatches
8,192 and entropy coefficient .005. Minimum attempts increase from 98,304 to
1,048,576; plateau patience from 49,152 to 262,144. Review budget: 2,097,152.
These are compute/stopping choices, not guarantees of convergence. Evaluation
still occurs every four updates. A fixed-scenario hit is not a robustness rate.

## Likelihood and reward accounting

At each decision, sample a bounded Gaussian jerk once and evaluate its likelihood
once. Intermediate 30 Hz transitions do not become additional PPO samples.
Their rewards are summed into the decision transition. With gamma=1 and
GAE lambda=1, the episodic return and terminal potential cancellation remain
unchanged, including success/failure inside a block and a final partial block:

`R_block = sum(r_control); sum(R_block) = shared_score - initial_potential`.

The terminal flag comes from the block's last observed control step, and its
validity mask comes from the decision start. Quarantined rows earn no extra
rewards or likelihood terms. Other discount settings are rejected for repeated
actions until duration-aware discounting is implemented. Missing timing fields
mean historical one-step actions. Saved checkpoints bind their control-step
count; loading them with a different cadence is rejected.

This is PPO with fixed action repetition, not a new adaptation algorithm or an
implementation of FiGAR's learned repetition policy. Temporal action abstraction
has established precedent in [Learning to Repeat](https://arxiv.org/abs/1702.06054).
[Colored Noise in PPO](https://arxiv.org/abs/2312.11091) separately motivates
temporally coherent exploration, but its colored-noise sampler is not used here.
The optimizer remains the clipped surrogate described in the
[PPO paper](https://arxiv.org/abs/1707.06347).

## Verification and interpretation

34 focused tests passed, two historical-fixture CUDA tests skipped because their
saved fixture is absent. Separate actual-M2 CUDA checks passed: full 2,048-row
rollout and optimizer update, behavior/learning log-probability parity, terminal
return accounting, checkpoint loading, deterministic rollout versus expanded
command replay, and bit-exact historical one-step collection. Explicit selected
MPPI command replay retains score 1892.4306902448288 under the unchanged objective.
The smoke update took 6.38 s on RTX 4080; smoke weights are discarded. Initial
random actions failed feasibility in 98.49% of attempts, so early failure is
expected and is reported rather than hidden. All 4,108 protected files passed
the preflight hash check. See `preflight.json` and `tests.xml`.

The progress dashboard reports policy cadence, fold and impact score alongside
the existing outcome/phase plots. Use PPO > Training progress and choose
"PPO M2 whip - sustained exploration - fresh policy". Rehearse policy can freeze
and replay its latest or best saved policy during training. The supervisor
independently replays the final best policy and attempts complete recovery/CSV
only for a modeled hit; failed recovery remains explicitly reported. Original
physical MPPI selection remains unchanged.

Native Windows Qt/VTK checks verified the progress bars, live saved log, and
freezing/replaying the 40,960-attempt policy while training continued. That early
policy missed; its preview is preserved, not presented as a learned whip. After
the small dashboard addition, 13 relevant tests passed again; 4,107 protected
files and all 285 frozen training source files remain unchanged. The dashboard
file's intentional change is separately recorded against its original hash.
Observed training GPU utilization was 97%; early throughput estimates roughly
one hour to reach the minimum budget, and up to about two hours at the review
ceiling. This is an estimate, not a promised completion time.

Judge the result by target contact, fold score, backward release and a replay,
not only its hit fraction or impact speed. Previous PPO: total 1247.61,
fold 2.72, drone forward velocity +1.40 m/s at contact. Selected MPPI: total
1892.43, fold 432.82, drone forward velocity -1.52 m/s. The legacy complete-wave
stage detector is not a validated style gate and also reports zero for MPPI.
This is a development experiment with two changed training choices, not a
controlled ablation establishing the sole cause of improvement or failure.

## Requested complete PPO recovery

User requested the same post-whip recovery for PPO as MPPI. Shared recovery
already existed; PPO Training progress now defaults Include recovery + CSV on.
Whip-only preview remains an explicit option. No training/reward/physics change.
Frozen best checkpoint at 1,859,584 attempts independently
reproduces score 1914.82511, modeled contact 1.090667 s,
backward drone velocity -1.199 m/s and forward tip velocity 6.421 m/s at contact.
Complete 10.23333 s command/recovery is saved at
`runs/rehearsals_pva/20260911-142454-400090-M2-ppo-1859584-with-recovery`.
Smooth braking/turn, return to [0,0,1.255], final 3 s hover use exactly the same
shared recovery helpers as MPPI. Original learned whip packets are identical;
full predicted drone/cable recovery and envelope checks pass. This is an
intermediate checkpoint while job 20260911-142454-400090 continues; read its
status before reporting. Its final supervisor review remains separate. No
selected physical MPPI/flight/model change. Read audit
`runs/audits/M2-ppo-recovery-20260911`; 15 focused tests pass. Recovery is appended
by the existing trajectory generator, not a phase learned by PPO. No physical
validation is implied. Hover 10 s before CSV and land afterward remain external.
