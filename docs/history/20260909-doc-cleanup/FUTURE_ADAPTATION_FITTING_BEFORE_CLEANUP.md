> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Next adaptation fit: convergence and essential checks

User instruction supersedes the historical fixed80/24-update protocol. Current M1 is complete and frozen for the new PPO run; do not refit it now.

Before the NEXT model fit, replace fixed NN budgets with a saved, deterministic practical plateau rule on the training-only rollout selection objective. Keep best weights, finite-gradient checks and a separate maximum-update safety limit. Report plateau versus safety-limit termination explicitly; never label the limit as convergence. Nominal least-squares solver convergence/boundary status and cable physics search limits must also be reported, rather than asserting global convergence from a finite grid.

Proposed next-fit defaults to implement: selection every10 drone updates /6 cable updates; relative meaningful improvement0.5%; minimum80 drone /24 cable updates; patience5 selection checks; safety ceiling2000 drone /240 cable updates. Use the existing regularized training-only score and preserve best weights. These are engineering stopping thresholds, not proof of convergence. Freeze actual resolved settings in the job protocol.

Default validation scope: source/clock/normalization checks; finite fit parameters/residuals and successful model loading; training/execution consistency; one coupled whip-only replay per take, M0 versus candidate with identical normalized initialization and recorded commands. Label these all-data diagnostics in-sample. No automatic five-fold refitting, component ablations, full recovery replay, or full-horizon gradient experiment. Those remain explicit research diagnostics only when a concrete issue requires them. Assess prospective new flights against the exact saved forecast BEFORE using them to fit the next model.

Implementation status: this instruction is recorded for the next fit. Existing historical fitter defaults remain frozen/unchanged; update the runner and stopping code before another fit. No extra validation or fitting should interrupt the presently authorized PPO adaptation.
