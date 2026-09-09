> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Direct-force MPPI: same rollout and reward as PPO

Open **06 MPPI Planner** after restarting the application. MPPI now optimizes a sequence of **30 Hz XYZ force actions**, using M1 by default. It does not optimize position spline points or train a neural policy.

The actual chain is:

1. Sample one normalized XYZ action at every 30 Hz control tick.
2. Use PPO's exact `physical_force` mapping: hover force plus scaled action, action clipping, nonnegative vertical force and total-force norm bound.
3. Run PPO's `plan_batch` on the shared-root force-controlled drone/cable model, including the cable residual.
4. Construct the same 30 Hz P/V/A reference with `frozen_reference` and `reference_packets`.
5. Call PPO's `execute_research_batch`: fitted loaded-drone model/NN → rotated attachment → DDER/cable NN.
6. Optimize its exact execution episode return. MPPI replaces only the actor's action selection. No predicted execution feedback enters the force plan.
7. Recheck the selected candidate, append the existing curved recovery/hold, and export the full 30 Hz FullState CSV.

The existing analytic recovery is unchanged and lies outside the whip objective. It is not a spline parameterization of the optimized maneuver.

## UI and saved results

**Plan setup** owns the new launch, model, horizon, batch size, iterations, temperature and action noise. **PPO reward settings** shows editable scalar weights from the selected seed's saved PPO configuration, plus the hit gate and strike direction. Edits belong to MPPI's config/job only. PPO checkpoint/config files remain unchanged.

Latest launch: tracked origin **[0,0,1.225] m**, target **[1.5,0,1.1] m**. Default model is `data/model_candidates/20260908-adp0-M1/model.json`. Default source is the flown PPO rehearsal `20260908-203914-039721` and its saved action/reward/termination contract. Defaults: 128 random candidates (+2 deterministic), 12 iterations, temperature 20, normalized action standard deviation .05, **one-second horizon (30 actions)**. For the source action scales, .05 corresponds to .10/.10/.08 N XYZ noise before clipping. Longer horizons are editable on a 30 Hz grid up to five seconds; beyond the saved seed forces, initialization uses hover, without stretching time.

The selected PPO is a historical one-second task: its `frozen_virtual_plan` cutoff/follow-through and time cost are preserved. In particular its saved time weight is **0.1 per second**, not the 10/s used in a different historical PPO objective. A source configured with `execution_success_or_timeout` uses that same existing PPO branch. No silent objective or stopping redesign is introduced by switching optimizers.

The inherited 45-degree tip-velocity direction gate still permits lateral approaches; it does not require the entire motion to remain in the XZ plane. The force representation alone does not impose a straight or planar whip.

**Rehearsal & export** defaults to results matching the current launch, target, model contents and direct-force representation. Editing launch/model removes incompatible results from this view and clears/disables the old preview/export. It does not move old geometry. A matching saved result is explicitly labeled with its run name; it is not claimed to reflect subsequent reward/search edits.

Choose **Saved runs — original coordinates** to deliberately inspect older results, including historical spline MPPI results. The banner identifies their original start and target; old arrays/CSVs are not rewritten. Only rehearsals that contain saved virtual forces are accepted as force initializers. The seed is a force guess, not an exported path to translate.

## Jobs and outputs

`planning/mppi_force.py` supplies batched fixed action sequences to PPO's unchanged pipeline. `planning/mppi_force_run.py` creates immutable jobs and exports results. The isolated worker is `tools/plan_mppi_force.py`. New jobs contain `mppi.json`, `ppo.json`, `task.json`, model assets/provenance and a source snapshot. They have **no `cem.json`, spline control points, spline coefficients or spline seed file**.

`actions.npz` stores the candidate normalized actions. `virtual_force_30hz.csv` stores the actual virtual force used by the simulator, **including gravity**. It is diagnostic simulator input, not a force-controller flight CSV. `fullstate_30hz.csv` is the complete real-controller command sequence. `rehearsal.npz` contains the frozen commanded PVA and predicted drone/cable trajectory. Result schema is `mppi_force_fullstate_30hz_v1`; ZIP export preserves exact CSV bytes and model assets. Adaptation Check can match the new forecast by the exact flown CSV hash.

The MPPI importance-weighted update is unchanged: fixed Gaussian prior centered on seed actions, shifted proposals, explicit prior/proposal density ratio and stable exponential weights. Every timestep/axis is an independent optimization coordinate; there is no interpolation through spline points. Deterministic mean/incumbent rows are excluded from importance weights. PPO's finite failure penalties remain part of the objective, so failed proposals are not silently replaced by the former spline objective or all discarded. Nonfinite scores are excluded; final export requires a valid complete reference/prediction and configured height limits.

## Verification and limits

24 distinct targeted tests passed across focused runs. GPU parity testing replays the retained PPO forces through direct-force MPPI with its original M0 setup: force equality within 2e-12 N, commanded FullState and predicted drone/cable positions within 1e-8 m of the exact saved PPO whip. Tests also check batched versus independent execution and unchanged source PPO reward/action/deployment configuration. GPU batch-size changes are not bitwise identical: return tolerance1e-5, reference tolerance1e-6 and final cable tolerance1e-5m cover the observed micrometre-scale differences. This is separate from the tighter saved single-case PPO parity check.

A bounded **16 random candidates × 2 iterations** M1 check at the new requested launch completed in **56.13 seconds**. It produced finite feasible candidates and a complete 11.2333 s CSV with an unchanged one-second whip. Selected return 28.7474; closest predicted tip distance **37.824 cm**, **no valid hit**. This is a pipeline/export check, not a converged maneuver or a recommendation to fly it. Its commanded/predicted-drone/predicted-cable sampled peak heights are 2.6140/2.4653/2.4859 m. Export/scoring prefix difference is zero. Exact force/FullState CSV and ZIP bytes were checked; 286 protected files remained unchanged.

Audit: `runs/mppi/20260908-M1-direct-force-check/verification.json`. Native Windows Qt/VTK setup, replay/rewind, force/PVA plots and CSV controls: `runs/audits/20260908-mppi-force-ui`. Tested on this Windows / RTX 4080 system. No new PPO training, model fitting or real flight was performed. Real execution remains fixed open-loop FullState playback after takeoff/settling; no aircraft sender or online MPPI controller is implemented.
