# Future fitting: convergence and essential checks

Current completed run: [M1-full to M2](../development/M1_TO_M2_ADAPTATION.md),
`runs/adaptation/M2-full-whip-v1`, 27.8 min, mixed combined results, not promoted.
Do not duplicate or restart. Both parent residuals were retained and updated; prior training
replay excludes historical held-out takes. The full pipeline uses practical
neural plateau stopping and verifies selected gradients before evaluation.
The older completed-M1/no-new-fit notes below describe the preceding decision.

Latest authorization: the complete [drone/cable/residual workflow](FULL_MODEL_ADAPTATION.md)
completed as continuation `runs/adaptation/M1-full-whip-v2`. Do not restart or
duplicate it. M1-full is registered for comparison, not promoted. This supersedes
earlier no-fit and scalar-only notes below. The
stopping, provenance and validation requirements remain applicable.

Current real trial: [M0/M1 first adaptation](../development/M0_M1_FIRST_ADAPTATION.md). One nominal
horizontal response gain fit completed at plateau and is not promoted after
held-out regression. No fitting is running; do not resume/retry automatically.
The scope was chosen from adaptation-only drone/cable diagnostics. Older statements
below about synthetic-only adaptation are superseded by that measured trial.

**Current whip workflow:** read [M0→M1 preparation](../development/M0_TO_M1_ADAPTATION.md).
It supersedes the historical normalized/adp0 launch details below. No real M1
fit is active; only synthetic software checks were run during preparation.

**Current experiment reset:** the repaired 145 g drone / 17 g cable workspace starts empty. Previous runs and data below are archived historical evidence. Read [NEW_SYSTEM_CHECK.md](../setup/NEW_SYSTEM_CHECK.md) and [HANDOFF.md](../../HANDOFF.md) before acting on old run instructions.

Fitting and PPO are currently stopped. The fresh PVA M0 stopped at full-whip cable
residual update 105 before plateau; the historical normalized M1 used for MPPI is
a separate frozen model. Do not resume either as part of documentation or replay.

For a newly authorized fit, use saved deterministic practical plateau stopping on
the training-only rollout selection objective. Preserve best/current weights,
optimizer and stopping history. The latest user-directed full continuation has no
routine neural update ceiling: use practical plateau, manual stop and numerical
checks. The old requirement for a fixed NN ceiling is superseded. Small nominal
least-squares searches retain their separate evaluation guard. Report plateau,
manual stop, solver termination and any ceiling as distinct outcomes. A finite
search or grid boundary is not global convergence.

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
See [paper handoff](../paper/PAPER_WRITING_HANDOFF.md) for research experiments beyond these
essential checks. Historical fitting recommendations are archived.
