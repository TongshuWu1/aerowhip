> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Offline MPPI with M1

**Superseded for new planning:** the user explicitly requested direct 30 Hz forces, exactly as PPO, with no position-spline maneuver. Read [MPPI_FORCE_PLANNER.md](MPPI_FORCE_PLANNER.md) for the current implementation and UI. The rest of this document records the preserved historical spline planner and its results.

Open **06 MPPI Planner**. MPPI replaces the removed CEM page; Adaptation Check remains seventh. Saved CEM artifacts are preserved, but the application exposes only MPPI for offline optimization. MPPI defaults to the fitted M1 model and the actual flown PPO rehearsal as a spline seed. No PPO inference or training is performed. M0 remains the PPO workspace model; MPPI's model selection is independent.

Choose the model JSON, seed, launch, target and search settings on **Optimize spline**, adjust the independent objective under **Task & rewards**, then click **Optimize with MPPI**. A completed result opens **Rehearsal & Export**, including native 3D playback, PVA/tracking plots, exact CSV and a portable trajectory ZIP. Existing selected policy and original deployment forecast are preserved.

Settings live in `config/mppi.json`; jobs live in `runs/mppi`. Seed refresh keeps edited launch coordinates. Each job freezes model assets, both residuals, objective, hit criteria, launch, seed provenance and source files. `model_provenance.json` identifies the model actually used instead of inferring it from the seed. `mppi.json` records MPPI settings; `cem.json` is an identical compatibility copy for the shared spline execution worker, not a second optimizer. Saved result schema is `mppi_fullstate_30hz_v1`. Original/current Adaptation Check ghosts are still matched by exact flown CSV, never regenerated using M1.

## Algorithm and scope

This is an **offline spline-parameterized MPPI variant**, based on the information-theoretic importance-sampling formulation of [Williams et al.](https://arxiv.org/abs/1707.02342). It optimizes the entire maneuver before export. It is not receding-horizon onboard feedback, and makes no guarantee of outperforming CEM for a given objective.

Let `z` contain the free quintic spline coefficients and proposed duration. The first three coefficients impose the fixed settled hover boundary. The Gaussian prior is centered at the refitted seed `z0`; proposals are centered at the current `mu`, with the same fixed diagonal covariance `Sigma`. For each independent random candidate:

```
log_weight = score(z)/temperature
             + log N(z; z0, Sigma) - log N(z; mu, Sigma)
weight = softmax(log_weight)  # infeasible candidates get zero weight
mu_new = mu + sum(weight * (z-mu))
```

The prior/proposal density ratio is included. This is not CEM with a renamed elite average. Covariance stays fixed; no elite selection enters the update. The search retains up to 32 distinct best evaluated candidates for final checks. Two deterministic candidates (mean/incumbent) are evaluated in addition to the configured random population, but excluded from importance weights. If no random sample is feasible, the mean remains unchanged and effective sample size is zero. If every candidate is infeasible, the job fails clearly and exports nothing.

Noise is sampled in spline coordinates, producing correlated smooth motion in time. Duration is a latent Gaussian coordinate decoded to a bounded 30 Hz grid; density ratios use the latent value, not the clipped duration. Thus this is not an exact continuous-time stochastic-control solution. Position noise, duration noise and temperature are tunable. Higher temperature spreads weight across more candidates; lower temperature concentrates it. The fixed seed prior regularizes changes from the starting maneuver. A very small feasible fraction means few useful proposals, regardless of optimizer.

History records best score, nominal candidate hit/feasibility fractions, effective sample size, maximum sample weight and reference/prediction feasibility. These are search diagnostics for one nominal initial condition, not held-out policy success rates.

Defaults are 128 random candidates (+2 deterministic), 12 iterations, temperature 20, 12 spline points, 5 mm position noise and 20 ms duration noise. Latest requested launch is **[0,0,1.225] m**, target **[1.5,0,1.1] m**. These apply to new MPPI plans; PPO and saved results retain their own coordinates. The simulation verification below used the earlier launch and is not a validation of this new target.

## Model, objective and execution

Every proposal uses analytic command P/V/A at 30 Hz → selected fitted loaded-drone model/NN → rotated attachment → DDER/cable NN at 150 Hz. M1 includes the extended cable residual and smooth curvature-frame model; no physics is replaced. Whip reward and hit gates reuse the existing explicit spline objective; PPO reward is not used.

Recovery prediction error is not a fitting acceptance requirement or optimization objective for this whip-focused task. Complete command and predicted-motion feasibility checks still apply before export. Whip ends at the predicted valid hit rounded up to a 30 Hz boundary, otherwise at its proposed duration, followed by the existing curved recovery/hold. Failed recovery checks are retained in `recovery_rejections.json`. A feasible predicted miss may be exported and is labeled as a miss; completion does not imply successful hitting.

The real sequence remains takeoff → settle 10 s → execute the complete fixed 30 Hz FullState CSV → land. This code does not send aircraft commands. Save the exact CSV and its prediction with the five paired adp1 recordings, evaluate M1 on those flights before fitting M2, and report actual virtual-target error separately from model prediction error.

## Verification on this computer

44 targeted tests passed, plus native Windows Qt/VTK setup/replay/rewind/PVA and two RTX 4080 integration runs. The smaller-noise run `runs/mppi/20260908-M1-mppi-local-check` used 128 random candidates × 8 iterations, completed in 75.75 s, and selected a **predicted valid hit at 0.91333 s, 2.416 cm closest tip distance**. Complete CSV is 11.13333 s; whip ends at 0.93333 s. Maximum sampled commanded/predicted-drone/predicted-cable heights are 2.5663/2.4526/2.4219 m. Exact whip-prefix consistency, continuous command envelope, complete prediction, CSV/ZIP byte equality and M1 source hash were checked. 289 protected files remained unchanged. No real flight, robustness assessment or policy training was performed.

The earlier 25 mm / 80 ms noise run retained a simulated miss with very few feasible proposals. It remains saved for comparison. The observed improvement also changes exploration size and sample budget, so it is not evidence that MPPI intrinsically outperforms CEM. Reports live in each job's `integration_check.json`, and UI/native scene images in `runs/audits/20260908-mppi-ui-final`.
