> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Farther target: dynamic strike reward v1

User authorized a separately saved reward experiment and fresh PPO training on 8 September. The previous farther-target run stopped by user request at 7,168 attempts. It remains stopped. This is a development experiment, not an established improvement or independent paper result.

## Task and unchanged model

- Initial attachment [0, 0, 1.5] m, target [1.5, 0, 1.4] m, both sampled within their existing 5 cm balls.
- Five seconds **maximum preparation plus strike**, 150 possible force actions at 30 Hz. No prescribed 1–1.5 s hit deadline or scripted phase switch.
- Existing first-contact gate: tip first within 5 cm, world tip velocity projected along +X at least 4 m/s, world velocity angle at most 45 degrees.
- Existing offline force planning, native 30 Hz FullState reference, fitted M0 drone and cable with both residuals. Planning cable is hanging; execution prediction uses the existing perturbed truth state. No future physical feedback enters the plan.
- Existing reference-feasibility limits and masses (157 g drone, 18 g cable) remain unchanged. No new fit or physical parameter change.
- Existing preplanned first-contact/horizon cutoff and follow-through remain. FullState export still appends gentle recovery; recovery is outside the PPO objective.

## Reward

For each executed prediction, let `d` be tip-to-target distance, `D` attachment displacement from its start, `vw` world tip velocity projected along the desired direction, and `vr` `(tip velocity − attachment velocity)` projected along that same direction.

The bounded speed function is the existing normalized sigmoid:

```
S(v,c) = clamp((sigmoid((v-c)/(c/4)) - sigmoid(-4)) /
               (0.5 - sigmoid(-4)), 0, 1)
Q = exp(-0.5 * (d / 0.30)^2) * S(vw,4) * S(vr,6)
P = clamp((initial_tip_distance - d) / initial_tip_distance, 0, 1)
A = max(0, distance(initial_attachment, target) - cable_length - 0.05) + 0.25
C = log(1 + (max(0,D-A)/0.50)^2)
```

`A` is calculated separately for each sampled initial state/target, using the final task radius (not a curriculum radius). At the nominal target, it is **0.75083 m**: approximately 0.50083 m required by the geometric reach bound plus a 0.25 m preparation margin. The reach bound assumes a fully extended cable; it is not a guarantee of dynamic reachability. The margin is a reward-design choice. This allowance is isotropic, with a soft penalty outside it; it is not a flight workspace limit.

| Component | Weight |
|---|---:|
| Improvement in best normalized progress P | +80 |
| Improvement in best combined strike quality Q | +120 |
| First valid hit | +250 |
| Integral of C over the scored maneuver | −1 per second |
| Increase in maximum C | −10 |
| Planned execution time | −0.2 per second |
| Another cable marker enters first | −50 |
| Invalid first tip contact | −25 |
| Numerical failure, invalid reference or invalid pose execution | −500 |

Timeout, return-at-hit, release-at-hit, terminal displacement and recovery weights are zero. No impact-angle shaping is added. The hit gate retains its original definition, so hit rate remains interpretable; the new reward prefers dynamic strikes among hits. A carried-tip hit can still pass the gate, but earns no combined speed quality if relative forward speed is zero.

Best-so-far shaping has a finite total budget: repeating an unchanged approach or swing earns no additional progress/quality reward. A root retreat under a stationary tip earns zero Q because the world-speed factor is zero. Translating a motionless cable earns zero Q because relative speed is zero. Speeds above the 4/6 m/s shaping caps earn no extra credit. The largest positive return is 450 before costs, and an invalid execution loses its hit bonuses; the 500 failure penalty exceeds the entire shaping budget.

The combined quality encourages useful relative tip motion; it does **not** prove a travelling whip wave or rule out all pendulum-like swings. Inspect cable shape and marker speed evolution before making a whipping claim.

## Training and records

Fresh seed 656, no checkpoint/optimizer resume or scripted force prior; existing PPO architecture, exploration, learning rate and optimization settings. 1,024 CUDA environments, 500,000-attempt review budget, deterministic validation on the same 256 development scenarios after each batch. No curriculum. Built-in plateau stopping uses the user's 20,000-attempt patience/minimum and the existing 20-evaluation window, 2-point/1% improvement tolerance. A plateau at low success is not evidence of a good policy.

Candidate configuration: `config/experiments/20260908-dynamic-strike-v1/`; `previous_workspace/` preserves the settings before this change. Applying the candidate updates only the native 30 Hz workspace. Each production run owns its config, model weights and source snapshot. Historical 20 Hz configs/checkpoints are unchanged; new reward keys default to the previous equations when absent.

Attempt records now include world tip, attachment-relative tip and attachment forward velocities at valid impact, impact displacement, and the travel allowance. Failed attempts have NaN for valid-hit measurements. Validation records/UI include mean relative-tip and attachment forward speed at valid hits. Compare accuracy, hit speed, relative speed, displacement and failures; old and new total rewards are different objectives and must not be compared directly as performance scores.

The saved live scene contains the actual stochastic training batch for this run. Isaac is still only its renderer; the calibrated CUDA model remains the training physics. No Isaac physics migration or real flight sender is involved.

## Checks

`tests/physics/test_dynamic_strike_reward.py` checks carrying/retreat counterexamples, bounded monotonic speed credit, non-repeatable quality rewards, unchanged world hit gates, per-scenario reach allowances, exact legacy displacement costs, impact records and invalid settings. Existing physics/scoring/UI/research tests remain applicable.

`tools/check_dynamic_strike.py` checks a full five-second stochastic rollout and PPO update on the GPU, reward-component/Monte Carlo accounting, scene publication, and a separate conservative approach probe through the same complete model. The approach probe is not installed as a policy prior and does not establish that the farther-target whip will succeed. Results are saved under `runs/audits/20260908-dynamic-strike-v1/`.
