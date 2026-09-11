# Deployment experiment protocol

The current protocol is the [one-day operator runbook](LAB_RUNBOOK.md).

User decisions on 11 September 2026:

- Keep existing M0 and preliminary recordings; no new M0 fit.
- Collect five M0 whip takes, fit new M1; collect five M1 takes, fit new M2.
- Finish with five interleaved M0/M2 pairs: 20 new whip executions in total.
- Use continuous minimum 3D target distance, with prediction error separately.
- Use one target. PPO, additional targets and component ablations are outside
  this day's collection.
- Export CSVs only; actual aircraft execution uses the external lab program.

The earlier research release candidate proposed fresh preliminary data, a
same-capacity M0 and up to 90 final flights. That proposal is superseded for this
deployment study. The preserved M0 capacity distinction remains explicit in
[reproducibility notes](REPRODUCIBILITY.md).
