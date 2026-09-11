# Two-target continuous-whip development trial

Historical technical record. The two-target planner, rehearsal and audit outputs
named here were removed in the user-authorized 11 September 2026 cleanup.
The numerical descriptions below record the original development review; its
output paths are no longer available in this checkout. The current paper uses
[one target and continuous distance](PAPER_EXPERIMENT_PROTOCOL.md).

The user requested one continuous sweep through two distinct targets, not two
separately initialized strokes. The original selected single-target M2 flight
remains `20260910-211435-608306`; these are separate development trials.

## Frozen initial experiment

Before searching, centers were fixed to T1 `[1.10,-0.15,1.00]` and
T2 `[1.40,+0.15,1.00]` m: 42.43 cm apart, with nonoverlapping 5 cm spheres.
Tracked-origin launch is `[0,0,1.255]` m, level hover, zero initial cable motion.
Both trials use full frozen M2-frozen-refit-v1, including both residuals,
512 CUDA samples, four proposal families, a complete 1.5 s forecast, and the
same editable original command seed/baselines. Physics, original flight
selection, data, fitting and controller settings are unchanged.

`ordered_two_target_v1` uses swept tip-sphere intersections at the existing
150 Hz physics rate. T1 is latched without ending or resetting the simulation.
T2 counts only after T1; entries within one tick preserve their fractional order.
Overall success ends the whip after T2, subject to existing feasibility checks
through the next command boundary. A visit to T2 before T1 receives no credit.
Contacts do not apply collision forces. This is a virtual-target task; two
physical impacts, particularly their effect on the subsequent cable motion,
are not validated by this model. Ordered contacts alone do not establish a
particular wave shape: replay and motion between entries must also be reviewed.

## Results (11 September 2026)

| Trial | Updates / search time | T1 | Closest T2 center | Outcome |
|---|---:|---|---:|---|
| `20260911-133034-712876` | 35 / 116.09 s | 1.1052 s, 4.99 m/s forward | 35.41 cm | 1/2 hits |
| `20260911-133519-124817` | 28 / 90.29 s | 1.1472 s, 2.69 m/s forward | 23.04 cm | 1/2 hits |

Both stopped by score plateau. Neither found the requested successful sweep.
This does not establish infeasibility: the search uses local variations of
single-target command seeds. No second contact or hitting velocity is invented.
The second trial's nearest T2 approach is at 1.4674 s, with XYZ error
`[-0.07387,-0.13133,+0.17430]` m: short in X/Y and above the target.

The first reward extension allowed a strong T1 impact to outweigh progress
toward T2. The second freezes `task.two_target_reward=completion_v2`:
`p_i = exp(-(d_i/0.35)^2)` and sequence credit
`1000*(p_1 + I[T1 hit]*p_2)`. Existing style/effort terms contribute at 0.1
of their saved scale. The impact bonus is the average of the two saved
`1600*v_i^2/(16+v_i^2)` forward-speed bonuses, available only after both hits.
Feasible complete two-target successes still rank before misses. These weights
are development choices, not paper-established constants. Old frozen source
and v1 semantics are retained; this is not a matched model-adaptation result.

## Replay and verification

Former replay (removed): `runs/rehearsals_pva/20260911-133519-124817-M2-two-target-whip`.
It contained the exact new PVA CSV, full independent forecast and checked smooth
recovery (10.2333 s total), explicitly labeled **1/2 hits**. It was saved for
review and was never selected as a successful two-target flight. The app's Live 3D search and
Rehearsal and export show both target markers and separate contact status.

Former audits (removed): `runs/audits/two-target-whip-20260911` and
`runs/audits/two-target-completion-20260911`. Fifty-four focused tests pass;
CUDA/eager/branch state checks pass including physical continuation after T1;
the original single-target score differs only by 3.7e-10. Independent NumPy
intersection timing matches the production replay. Full export verifies
command/drone/cable prefix parity and complete recovery. Native Qt/VTK live
and rehearsal playback show both targets; 2,559 protected files are unchanged
in the second audit (including the first trial), with no fit or physical flight.

Single-target readiness remains qualified: its command/replay package is
checked, but the three existing M2 real flights did not enter the virtual target
sphere. Clean-paper release conditions remain in PAPER_PIPELINE_AUDIT.md and
PAPER_EXPERIMENT_PROTOCOL.md.

The user subsequently asked about a different Y, then confirmed the current
positions are fine. The proposed T2 Y=+0.30 trial was cancelled after bounded
seed preflight, before any optimizer job was prepared or started. Its former audit (removed) was
`runs/audits/two-target-y030-20260911`; it is not a completed MPPI trial.
