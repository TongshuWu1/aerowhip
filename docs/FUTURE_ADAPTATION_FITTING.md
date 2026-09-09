# Future fitting: convergence and essential checks

Fitting and PPO are currently stopped. The fresh PVA M0 stopped at full-whip cable
residual update 105 before plateau; the historical normalized M1 used for MPPI is
a separate frozen model. Do not resume either as part of documentation or replay.

For a newly authorized fit, use saved deterministic practical plateau stopping on
the training-only rollout selection objective. Preserve best/current weights,
optimizer and stopping history. Keep finite-gradient checks and a separate maximum
update safety ceiling. Report plateau, manual stop, solver termination and ceiling
as distinct outcomes. A finite search or grid boundary is not global convergence.

The current implementation includes stopping support in
`experimental_data/plateau.py` and the PVA/bootstrap fitting paths. Inspect the
new job's actual resolved protocol rather than assuming historical proposed
80/24-update budgets or thresholds apply to every stage.

Essential validation: source/clock/normalization integrity; finite parameters and
residuals; model loading; training/execution consistency; and one coupled whip-only
replay per take comparing baseline/candidate with identical initialization and
recorded commands. Label all-data diagnostics in-sample. No automatic expensive
fold refitting, full-recovery sweep or ablation campaign without a concrete request.

Assess prospective recordings against their exact saved forecast before using
them for another model update. Preserve old data roles and source snapshots.
See [paper handoff](PAPER_WRITING_HANDOFF.md) for research experiments beyond these
essential checks. Historical fitting recommendations are archived.
