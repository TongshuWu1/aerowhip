# Development reward tuning

Objective: a fast valid cable-tip impact, followed by PID recovery, with limited
maximum drone excursion. Tip speed is an impact proxy; this simulator does not
measure contact force or target deformation.

The pre-tuning run `20260905-202727-652710-seed651` was stopped cooperatively at
40,960 attempts. Its checkpoint and histories are retained. Candidate runs use
isolated immutable snapshots and start from the same saved weights. No real
flight, physical-fitting update, or protected recording is involved.

Historical comparison: `ea233e4:PPO_WHIP_HANDOFF.md` describes D50 at 93.75%
validation success, median directed tip speed 5.10 m/s and mean maximum drone
excursion 0.916 m. This was a different 84-input, closed-loop UAV controller,
not the current 79-input point-force/open-loop system. The progress/speed/hit
weights were already broadly similar. Historical pure attachment-compensated
progress and blended-progress experiments failed; they are not imported as a
successful recipe.

New diagnostics expose planning distance and maximum drone travel separately
from actual execution. Previously a refused plan reported the stationary
execution state's initial tip distance, despite receiving a nonzero planning
reward. Reward components now sum to the total, including time corrections,
numerical failures and PID recovery costs. Successful impact speed is recorded.

The shared tuning setup uses 1,024 environments, learning rate 0.0003, Adam
reset, minibatch 4,096 and 128 paired development-validation scenarios. These
are acquisition experiments, not independently replicated paper results.
Simulator, force limits, 7 s horizon, 5 cm / 4 m/s / 45 degree tip-first gate,
initial-state uncertainty and recovery duration are retained.

Candidates:

- Reference: saved reward, with the shared tuning optimizer settings.
- Approach60: increase progress from 20 to 60 only.
- Balanced: approach60, proximity width 0.30 m, hit bonus 200, explicit time
  penalty 0.1/s, displacement integral 0.5/s, terminal displacement 20,
  return and release bonuses zero.
- Compact: balanced plus maximum-displacement weight 15. This cost increments
  only when maximum excursion increases and sums to
  `-15 log(1 + (maximum displacement / 0.35 m)^2)`. It applies to both hits and
  misses; returning cannot erase the cost. Its default is zero for compatibility.

Candidate comparison must use valid-hit and hit-plus-recovery rates, successful
impact speed/time, and maximum drone travel. Total returns with different reward
coefficients are not comparable. Plots preserve raw validation points and will
not be labeled as final paper evidence.

The balanced candidate's starting policy had zero valid plans on 128 scenarios,
median planned tip distance 0.094 m and mean maximum planned drone travel
1.233 m. Its reward components were +55.73 progress, +1.32 strike quality,
-19.53 non-tip-first, -2.16 travel integral and -0.26 time. Increasing progress
alone therefore risks reinforcing translation without a valid impact.

Artifacts: `data/reward_tuning_20260905`; preparation script
`tools/reward_tuning_experiment.py`; plots `tools/plot_reward_tuning.py`.
## Screening observations

Balanced (4,096 additional attempts) and compact (4,096) produced no valid
deterministic validation hits. Compact reduced mean planned maximum travel to
1.029 m, but its final validation contacts were all non-tip-first. The
deployment-aligned pilot was stopped after 2,048 additional attempts with no
validation hits. These are adaptive early stops, not equal-budget final results.

Two mechanism changes are implemented as explicit configuration options:

- `reward.directed_speed_shaping_reference="world"`: a backward-moving tip
  no longer receives forward-speed shaping simply because the attachment is
  retreating faster. The actual hit gate was already world-referenced.
- `deployment.require_predicted_success=false`: execute each finite predicted
  attempt once, with its frozen first-contact/horizon cutoff; an unsuccessful
  prediction no longer means skipping plant execution. Numerical/state-limit
  failures remain refused. Predicted hit rate and execution rate are separate.

A paired planning-only diagnostic (1,024 samples per condition) found 0 valid
hits with the original XYZ exploration, 2 with Y latent std 0.15 and 3 with
Y std 0.05. Reducing XZ noise as well reduced wandering. These tiny hit counts
motivate acquisition changes but do not establish a statistically reliable gain.

## Explicitly labeled searched initialization

An offline CEM feasibility search used 8,192 nominal attempts, seed 617, a 2 s
search horizon, six interpolated X/Z action knots, and the unchanged final hit
gate and force limits. Its best nominal sequence hit at 0.79 s with 0.818 m
maximum drone displacement. It is **not PPO**. Search effort must be counted
separately in comparisons, and zero PPO attempts must not imply zero data cost.

The final searched sequence was truncated to a 1 s maximum attempt and evaluated
on 128 development scenarios (seed 90651), with 10 s PID recovery:

- 52/128 valid impacts (40.625%); nominal 28/28.
- Mean successful impact speed 5.685 m/s, time 0.791 s.
- Mean maximum actual drone travel, including recovery: 0.815 m.
- Hit plus full cable-settling recovery: 28/128 (21.875%); nominal 28/28.

These are development simulation results with provisional uncertainty ranges,
not hardware results or a final held-out claim. The earlier intermediate seed
evaluation used a different searched sequence and 2 s cap; its 50% hit rate is
not a controlled horizon-only ablation. Its miss tails caused excessive travel.

`20260905-214831-827697-seed651` initializes the existing action-prior mechanism
from the final sequence and trains a state-dependent PPO residual. PPO starts
at zero attempts, with latent std [0.05, 0.02, 0.05], learning rate 3e-5,
entropy weight 0.001, 1,024 environments, and 8,192 requested attempts. The
whole force sequence is still planned from the initial state and executed
without strike feedback. No contact limits were relaxed. Result selection is
pending; root task/reward defaults have not yet been replaced.

The impact-speed diagnostic was corrected to record at the successful physics
step, rather than at the episode horizon. Earlier pilot speed values are missing,
not zero. Reward and hit decisions are unaffected; an early-hit regression test
now covers this. New training runs archive source files and a package manifest.

At 3,072 PPO attempts the original-cutoff policy reached 75/128 valid hits
(58.594%), mean successful impact speed 5.666 m/s and time 0.792 s. Its complete
10 s cable-settling recovery rate remained zero. It was preserved and stopped
for an explicit execution-schedule continuation, rather than silently editing
the running experiment.

A paired diagnostic of the 2,048-attempt checkpoint found 65/128 impacts with
all uncertainty, 67 with exact initial state, 100 with exact force gain, 117
with matched physics, and 123 with both exact state and physics. This is a
strike-only development sensitivity test, not a recovery evaluation or proof
that the provisional uncertainty distributions represent hardware.

With the same checkpoint and scenarios, adding a **precomputed** 20 ms hold of
the final strike force improved hits from 65 to 82/128; 40 and 60 ms also gave
82. This motivates `deployment.strike_followthrough_s=0.02`. The original
prediction still terminates at its first contact, and there is no extra actor
query, repeated swing, or flight feedback. The final force is held for two
physics steps, capped by the 1 s attempt limit, before PID takeover. A zero
setting exactly preserves historical execution. Live and training compilation
share this implementation and have a parity test.

Continuation `20260905-220217-115520-seed651` starts at 3,072 PPO attempts and
requests 4,096 additional attempts with this option. Root defaults are still
unmodified pending its result. The full live-flight/follow-through suite passed
18 tests, including the compiled-physics parity check.

The follow-through continuation started at 103/128 impacts (80.469%), with
0.848 m mean peak travel. At 5,120 lineage PPO attempts it reached 109/128
(85.156%), with 0.869 m mean peak travel. The following update fell to 105/128
and increased travel; the earlier policy remains available in the immutable
validation journal.

A paired **15 s recovery** check of the starting 3,072-attempt checkpoint
recovered all 128 scenarios, preserving exactly the 103 impacts and peak travel
from its 10 s evaluation. The original zero recovery rate was therefore a
settling-deadline issue. The 15 s allowance is an explicit protocol change,
not a relaxation of the spatial/speed settling criteria. Live flight already
continues PID until the cable settles; no strike feedback is added.

Final selection protocol, before inspecting the reserved seed 190652:

1. Choose the highest development valid-impact rate with mean peak actual
   travel <=1 m and mean successful hit time <=1 s; break ties by smaller travel.
2. Allow 15 s recovery and perform one guarded continuation update. Reject
   regressions in development impact/plan/recovery success rates; preserve the
   accepted checkpoint if an update is rejected.
3. Freeze the resulting checkpoint and evaluate 512 independent scenarios using
   the reserved seed once. Do not tune against those outcomes. Report nominal
   and varied cases, impact speed/time, peak-travel distribution, and recovery.

This is a development result with one training seed, searched initialization,
adaptive pilot stops, and a changed execution schedule. A paper comparison
needs explicit accounting of search and discarded pilot costs, independent
training seeds, and a preregistered final experimental protocol.

## Frozen result and publication

The final guarded run `20260905-221711-285892-seed651` rejected its one proposed
update and retained the selected actor. Its reserved 512-scenario evaluation
(seed 190652) produced 427 valid impacts (83.398%), 100% recovery, mean successful
impact speed 5.652 m/s and time 0.793 s. Mean peak drone travel was 0.872 m,
95th percentile 0.912 m, and the largest was 0.931 m. All 128 nominal cases hit;
299/384 varied cases hit. There were no numerical failures.

No tuning followed this evaluation. The tested reward, initialization, optimizer
guard, 1 s horizon and 15 s recovery defaults were published. Previous root
defaults are preserved in `data/reward_tuning_20260905/defaults_before_publication`.
No saved run configuration or fitted model was replaced. The selected run is
named **PPO · selected fast single strike** for live replay. The full result,
counts and limitations are in `docs/history/REWARD_TUNING_RESULT_20260905.md`.
