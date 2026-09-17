# M0_v2_002: four completed whips

All four complete moving command sequences match the accepted M0 CSV. Native Motive data and logger/event rows are preserved. Original files were not modified.

All four pass the existing strict one-second initialization check and have valid vehicle tracking. No TF gaps occur during the first 1.5 seconds of any whip. Missing/geometry-invalid cable samples remain gaps and are listed below; this is review, not automatic training acceptance.

| Whip | Invalid marker samples (c1–c10), first 1.5 s | Tip valid | Alignment residual |
|---|---|---:|---:|
| 1 | [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] | 100.00% | 0.214 mm |
| 2 | [1, 0, 0, 0, 0, 0, 1, 1, 1, 0] | 100.00% | 0.247 mm |
| 3 | [1, 0, 0, 0, 0, 0, 0, 6, 0, 1] | 99.33% | 0.250 mm |
| 4 | [0, 0, 0, 0, 0, 0, 1, 1, 0, 0] | 100.00% | 0.262 mm |

Offsets are estimated separately from measured TF and Motive positions, excluding logger gaps. They are not physical response delays or verified clock synchronization. There is no physical target. Keep all four repetitions grouped as one recording/battery. No model fitting was performed.

In Flight comparison > Flights by model, refresh and select this M0 session, then choose Take 001–004. In Rehearsals > Recorded takes, refresh to select the individual raw recordings.
