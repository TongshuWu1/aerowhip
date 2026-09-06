# PPO readiness audit — 5 September 2026

Decision: do not launch a million-attempt paper run with the current defaults.
The execution architecture supports an initial-state-only strike, but baseline
selection and training design need resolution before committing that budget.
This audit does not apply a model or alter training defaults.

## Findings before a long run

1. **The calibrated candidate is not active.** `config/baseline.json` still
   selects `20260905-051517-500223`: EI 4.02975e-5, Cb .001026632,
   zero external drag, three substeps, attachment [0, 0, -.055] m.
   The previously recommended candidate has EI 2.83994e-8, Cb 3.75231e-5,
   drag .3/s, twelve substeps and lateral offset [6.655, -12.874] mm.
   A fresh GUI run snapshots the active model. Merely selecting or fitting a
   candidate does not activate it. Continuing an existing run inherits its old
   model even after a new baseline is applied.

2. **The actor is restricted to X/Z actions.** `stochastic_action_indices=[0,2]`
   masks both sampling and mean outputs. With no action prior, Y is identically
   zero, not just noise-free. This is a legitimate planar-task ablation, but
   does not establish full 3D control. The deployment distribution nevertheless
   introduces lateral initial-state and estimation perturbations.

3. **The update guard can block acquisition.** `guarded_update_is_acceptable`
   requires displacement-cost integral to increase by no more than .2 s,
   alongside return and success conditions. A synthetic regression check with
   success 0→5%, joint success 0→5%, return -5→10 and cost 0→.21 is rejected.
   This proves the acceptance rule, not that a particular training run has
   stalled. Refused plans report zero executed displacement, so discovery of
   executable moving plans can encounter this constraint. Repeated rejection
   can halve learning rate from 3e-5 to the 1e-6 floor after five decays.
   Recommend disabling the refinement guard for fresh acquisition and retaining
   deterministic validation/best checkpoints; evaluate guarded refinement as a
   separate experiment. Do not silently call this guard ordinary PPO.

4. **Initial cable coverage is narrow.** `sample_batch` constructs a straight
   cable, perturbs its tilt by at most 1 degree and adds rigid angular motion.
   The actor observes all 12 node positions/velocities, but is not trained on
   arbitrary bending modes or reconstructed recording states. This supports a
   near-hover initial-condition claim. Broader full-cable-state generalization
   requires a length-consistent training state distribution and separate checks.
   `training.canonical_initial_state_fraction` is bypassed by deployment sampling;
   `deployment.nominal_fraction=.25` controls the active path.

5. **GPU-to-live numerical transfer remains a check for the chosen baseline.**
   PPO uses float32 CUDA with the experimental 32-iteration damping path; live
   compilation uses CPU float64. Existing live/reference tests pass, but they
   do not establish that a newly learned, threshold-sensitive strike preserves
   success and cutoff across both implementations. Compare identical saved
   states/forces and the hit margin before deployment. Twelve substeps also
   changes training cost relative to the old three-substep baseline.

6. **Live launch tolerances exceed the training estimation tolerances.**
   `LiveFlight.start_strike` permits up to .04 m node position drift and .15 m/s
   velocity drift during plan preparation. Training uses .002 m root estimation
   error and .01 m/s root velocity error, plus a small cable tilt error.
   These are different error models, and the launch thresholds are not validated
   by the current sampling distribution. Align them or explicitly test accepted
   preparation drift before treating live replay as a robustness demonstration.

7. **First valid hit is not necessarily first contact.** An invalid tip entry
   sets `episode_invalid_tip_entry` but does not terminate planning or block a
   later success; its configured penalty is zero. The swept sphere test also
   counts an interval already inside the target sphere, rather than requiring a
   new outside-to-inside crossing. Thus the compiler guarantees no tail after
   the first valid predicted hit, but not that the first contact was valid.
   If the experiment requires one physical impact only, invalidate the attempt
   on the first invalid contact and test that contract before long training.

## What the implementation does correctly

- Clipped on-policy PPO, normalized advantages, separate actor/value optimizers,
  GAE, gradient clipping and KL stopping are present. A 4,096-observation check
  found sample/evaluate log-probability agreement within 3.34e-6 at initialization.
- During compilation, policy observations come from a private nominal rollout
  initialized once. They are not real strike feedback. The resulting force
  sequence and cutoff are frozen before the independently perturbed plant runs.
- The compiler stops at its first predicted valid hit; unsuccessful plans are
  refused. Actual early hits or misses do not modify the force sequence. PID
  resumes at the planned cutoff. Tests cover this separation and mid-hold cutoff.
- Whole-plan reward, including recovery, reaches all planning actions with
  gamma=GAE-lambda=1. There is no implicit discount preference for an earlier hit.
- Angle has no shaping reward. Success uses world tip velocity: forward component
  at least 4 m/s, direction error at most 45 degrees, target radius .05 m, tip first.
  Speed shaping deliberately uses attachment-relative directed tip speed.
- Explicit elapsed-time penalty is 1 point/s, at most 7 points for a full plan,
  compared with a 100-point success bonus. Keep this low penalty initially.
- Success ends scoring once. Physical execution continues until the frozen cutoff
  and through recovery. Failed/refused attempts remain in aggregate denominators.
- Each GUI batch publishes current accepted-policy validation and a recording of
  its actual first trial. Rejected-update evaluation reuse is recorded explicitly.
  This is one visualized trial per batch, not five independently rendered trials.
- Fresh runs snapshot model, task and algorithm settings. Checkpoints and
  validation histories are retained. Resume is continuation, not an exact replay
  of an interrupted RNG/optimizer trajectory (optimizer reset is configured).

## Evaluation and paper interpretation

The 256 fixed validation scenarios influence update acceptance and checkpoint
selection. They are development/model-selection data, not an independent test.
The final 512-scenario seed is separate, but comes from the same provisional
uncertainty distribution. It cannot stand in for real-flight validation or the
protected experimental recording. Avoid repeatedly inspecting that final seed
while tuning pilots; reserve a fresh final seed for the locked experiment.

The UI curves are correctly labeled single-seed validation and exported without
smoothing. Paper evidence should add multiple independent training seeds, report
per-seed results and uncertainty across seeds, and show hit rate, hit+recovery
rate, return, planning refusals, numerical failures and wall time. With a guard,
also show accepted/rejected updates; an improving accepted-policy curve is a
selection outcome, not proof that every PPO update improved.

## Recommended sequence

1. Apply and freeze the reviewed physical candidate. Fix the task and initial-state
   scope: a near-hover planar demonstration or full 3D control. For the latter,
   enable all three actor outputs and establish cable-state sampling coverage.
2. Use fresh-acquisition PPO without the displacement refinement gate. Keep the
   current low explicit time penalty. Do not tune multiple reward terms at once.
3. Run a bounded diagnostic pilot before scaling: approximately 20,480–51,200
   attempts at batch 512–1,024 gives 20–100 collection/update rounds. This is a
   proposed diagnostic budget, not a guarantee that successful strikes emerge.
   Inspect actual trajectories, finite updates, exploration, plan discovery,
   deterministic success and any refusal/rejection plateaus. Measure throughput
   at the intended batch size; a small smoke test is not a million-run ETA.
4. Resolve pilot failures, freeze the protocol, then scale the budget and repeat
   across independent seeds. Compare the selected policy in GPU and live CPU
   dynamics before interpreting open-loop deployment performance.

At batch 2,048, a requested million attempts currently means 489 collection/update
rounds and 1,001,472 actual attempts (the final batch is not shortened), with at
most 70 planning decisions per attempt. Episodes and optimizer steps are not
interchangeable; report the actual attempted count and valid transitions.

## Verification

49 targeted tests passed, including PPO updates, reward gates, open-loop command
isolation, live recovery, validation recordings and short PPO/SAC launch snapshots.
Two existing Torch JIT deprecation warnings were emitted.
The GPU smoke triggered simulation failure limits in 12/32 exploratory attempts
on the active model and 11/32 on the recommended candidate. Both produced zero
hits in the 32-attempt collection and the 16-scenario deterministic evaluations
before/after one update; optimizer metrics and reward arrays remained finite.
This count does not distinguish NaNs from configured position/speed-limit
violations. Random exploration can leave the allowed region, so these counts
alone do not establish numerical instability. Add failure-reason diagnostics
before attributing failed exploration to the solver or tuning around it.
Collection took 41.75 s (active) and 167.86 s (candidate) at batch 32. This is not
a throughput benchmark at the intended production batch size.
`data/ppo_audit_20260905/contact_contract.json` records the synthetic first-contact
reproduction: invalid contact remains active at step 1, and success is counted
at step 2 while the invalid-contact flag remains set.
`data/ppo_audit_20260905/source_hashes.json` identifies audited code/config inputs.
`data/ppo_audit_20260905/smoke.json` contains bounded GPU checks on the active and
recommended models, using 32 training attempts, one update and 16 deterministic
validation scenarios per model. These checks establish execution/finite behavior,
not learning convergence or a calibrated uncertainty distribution.

PPO reference: Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
The deployment compiler and validation rollback are project-specific additions.
