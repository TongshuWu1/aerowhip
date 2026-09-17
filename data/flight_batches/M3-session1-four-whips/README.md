# M3_v2_001: 4 completed whips

All retained complete moving command sequences match the accepted M3 CSV. Native Motive data and logger/event rows are preserved. Original files were not modified.

Tracking and initialization checks are recorded per whip in report.json. Missing/geometry-invalid cable samples remain gaps and are listed below; this is review, not automatic training acceptance.

| Whip | Invalid marker samples (c1–c10), first 1.5 s | Tip valid | Alignment residual |
|---|---|---:|---:|
| 1 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] | 100.00% | 0.216 mm |
| 2 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] | 100.00% | 0.232 mm |
| 3 | [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] | 100.00% | 0.278 mm |
| 4 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] | 100.00% | 0.342 mm |

Offsets are estimated separately from measured TF and Motive positions, excluding logger gaps. They are not physical response delays or verified clock synchronization. There is no physical target. Keep repetitions grouped by original recording/battery. No model fitting was performed.

In Flight comparison > Flights by model, refresh and select this M3 session, then choose an available take. In Rehearsals > Recorded takes, refresh to select the individual raw recordings.
