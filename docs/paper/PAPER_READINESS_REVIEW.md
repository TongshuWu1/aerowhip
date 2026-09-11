# Paper readiness and release gaps

Updated 11 September 2026. Read the
[whole-system audit](PAPER_PIPELINE_AUDIT.md) and
[clean protocol release candidate](PAPER_EXPERIMENT_PROTOCOL.md).
The previous guide's exact bytes are preserved under
`runs/audits/paper-pipeline-audit-20260911/before/docs/`.

**Verdict:** the architecture and implemented staged estimator are defensible.
The methods can be written now. Reliable real interception and a clean causal
comparison of the update loop are not established, and clean collection is
not yet released.

| Element | Assessment |
|---|---|
| System contribution | Coherent: loaded UAV/cable identification, offline planning and measured refinement |
| Adaptation mathematics | Standard regularized simulation-error minimization; no new algorithm claim required |
| Full component implementation | Aircraft nominal/residual/attitude and cable physics/residual fitting exist |
| Joint fitting | Not implemented; not mandatory for the selected staged baseline |
| Fresh M0 consistency | Open: legacy preliminary bootstrap differs from full update path |
| Command and geometry | Internal chain tested; actual external sender/firmware and metrology still need verification |
| Launch initialization | Causal diagnostic estimator exists; live preparation/readiness procedure remains unverified |
| Numerical behavior | Focused gradient/replay checks and selected-M2 step refinement support this path; representative sensitivity still needed |
| Real target outcome | No observed 5 cm virtual entries in current 13 development takes |
| Model improvement | Useful matched-input averages, with per-take and retention regressions |
| Clean experiment | Designed in release candidate 1; missing fields must be closed before collection |

## Required next work

1. Make fresh preliminary-only M0 and subsequent full updates use the same model
   architecture, loss contract and numerical checks; reject scalar-only defaults.
2. Record and test actual sender/controller settings, reference points, target
   geometry and clock uncertainty.
3. Verify current-state launch readiness after planning, with calibrated thresholds,
   dwell, state age and timeout.
4. Complete representative fixed-command numerical sensitivity and task-observability
   checks. The selected M2 hit persisted at 8/16/32 cable substeps, but this alone
   is not global convergence or physical validation.
5. Freeze the manifest, data roles, target conditions, operating envelope, adoption
   margins, block order, collection budget and final analysis before fresh data.

The baseline remains staged full-model fitting, 512-sample/1.5 s offline planning,
fixed contact-first ranking and a soft contact-speed preference. The protocol
separates actual closest distance/entry from prediction error and wave-style
diagnostics. It does not require a PPO switch or another reward redesign.

## Verification in this audit

88 focused tests passed; one historical cold-seed fixture test was skipped.
Independent production M2 replay matches the saved tip forecast to `3.10e-13 m`
RMS. Finer cable steps retain the predicted entry; 8 versus 32 substeps differs
by 0.894 cm tip RMS and up to 2.72 cm instantaneously over 1.5 s.
Twelve usable launch histories show 0.73–8.83 cm drone start displacement and
0.010–0.136 m/s estimated tip speed. On a common 1.5 s real task window, all 13
takes still have no observed sphere entry.

646 protected evidence/configuration/core-source files are hash-checked unchanged.
No new fit, planner, model promotion, controller change or physical flight was
performed. Full artifacts and qualifications are in
[`runs/audits/paper-pipeline-audit-20260911`](../../runs/audits/paper-pipeline-audit-20260911/README.md).
