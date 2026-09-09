# Execution hit or five-second termination

The user's 8 September correction supersedes the previous planned-duration
time cost. The episode ends at the **first valid hit predicted by the fitted
drone and cable execution model**, otherwise at five seconds. Time cost is
`-10 * terminal_time_seconds`. With all other components equal, a two-second
successful strike earns 30 points more than a five-second successful strike.
The success bonus remains 250. Hover and recovery are outside this objective.

The previous implementation could keep executing/charging a virtual-model
plan after the fitted execution model had already hit. It could also stop
planning on a virtual cable contact even though predicted execution had not
succeeded. Increasing the coefficient alone did not correct that mismatch.

## Implemented contract

- Explicit opt-in `deployment.termination = execution_success_or_timeout` in
  the native 30 Hz workspace. Old frozen runs and absent-key behavior remain.
- The force policy still receives only its virtual planning states generated
  from the initial state and target. No actual or predicted execution feedback
  is fed into its action generation. Virtual contacts no longer stop candidate
  generation. The existing maximum horizon is five seconds.
- The separate execution model detects valid tip-first contact using the same
  distance, world-directed speed and angle gates. Each successful row freezes
  at that 150 Hz hit step, with no later reward, displacement accumulation or
  failure from its unused tail. Batch processing can continue for other rows.
- Invalid contacts retain their penalties and existing first-contact
  disqualification, but an unsuccessful finite episode continues to its time
  limit. Nonfinite states and invalid reference/pose/model computations remain
  terminal failures with the existing 500-point penalty. They never count as
  successful early termination. The declared timeout penalty is also applied.
- Failure checks are restricted to the prefix actually reached. A later bad
  virtual state, reference packet or pose cannot retrospectively erase a hit.
  Speculative invalid commands are sanitized only to keep batched predictor
  calculations finite; reaching one before a hit still fails the attempt.
- PPO includes only the control actions through termination, places the total
  return on the last included action and masks all later actions. The terminal
  action is marked done. Both reward-component and Monte Carlo sums agree.
- Live training replay and presentation exports freeze O/R and cable together
  at each row's actual terminal event. The shared batch timeline can still run
  to five seconds for other unfinished trajectories.

## Offline CSV and recovery

The rehearsal uses its own deterministic predicted hit time to shorten the
whip prefix, or exports five seconds of whip if there is no valid hit. It ends
that prefix at the next native 30 Hz packet boundary (less than 33.4 ms after
the scored event), completes those few unscored physical samples, then appends
the existing gentle braking, return and hold. Used P/V/A packets are preserved
exactly; no resampling or regeneration of their acceleration. Full CSV duration
can therefore exceed five seconds because it includes recovery.

This is a cutoff selected **offline from simulation**. Real CSV playback stays
open loop; there is no new real cable sensor, flight sender, online hit detector
or abrupt motor stop. If the real cable hits at a different time, the current
controller still follows the exported schedule.

## Saved continuation and checks

V2 `20260908-174509-301420-seed656` was stopped at 44,032 saved attempts and
81 cumulative successes. Its checkpoint is preserved. The new continuation
retains policy/value tensors and both optimizers, with new stopping history
because the terminal objective changed. The 20,000-attempt patience/minimum,
2,048-environment GPU batch, five-second maximum, M0 and both residuals, task,
action limits and PPO hyperparameters remain unchanged.

Settings and previous workspace:
`config/experiments/20260908-execution-hit-termination/`.
Audit: `runs/audits/20260908-execution-hit-termination/`.
Tests cover different hit times in one batch, timeout, invalid-contact
disqualification, failures before/after a hit, terminal PPO masks, frozen scene
poses and exact short CSV prefix/recovery continuity. The GPU audit checks
full-horizon deterministic and stochastic execution, an optimizer update and
a complete 30 Hz rehearsal. These are implementation/development checks on
Windows / RTX 4080, not independent physical validation or a claim that the
old checkpoint has already learned to strike earlier.
