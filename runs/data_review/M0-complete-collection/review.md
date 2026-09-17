# Complete M0 collection review

The collection contains 11 complete whips across three recording/battery groups. The original plan requested ten; all eleven are retained for review. Recording 001 repetition 4 remains excluded following the reported connection loss.

| UI session | Recording | Complete whips |
|---|---|---:|
| 1 | M0_v2_001 | 3 |
| 2 | M0_v2_002 | 4 |
| 3 | M0_v2_003 | 4 |

All eleven have all 146 moving CSV packets, valid native OptiTrack vehicle tracking, strict valid one-second initialization histories, and no logger TF gaps in the first 1.5 seconds of execution. Command snapshots agree with their event-log timestamps. The command events in the whipping intervals match the saved CSV.

Brief cable-marker gaps or geometry-invalid samples remain explicitly flagged. No samples were filled or moved. Recording 002 whip 3 has 99.33% valid tip samples and recording 003 whip 3 has 98.67%; all other tips have 100% during this interval. The most affected individual marker retains 96% coverage. These quality checks do not grant automatic training acceptance.

Synchronization remains estimated from the two measured position streams, not from matching real motion to the predicted trajectory. Native OptiTrack timestamps and 100 Hz data are preserved. The recordings have independent time origins; offsets are not physical drone delays. New segment alignment avoids interpolating across logger gaps.

There is no physical target, so geometric target distances are not physical hit results. The eleven repetitions come from three recordings and must not be treated as eleven independent flights. No model fitting or paper edits were performed.

All new takes were selected, loaded and rendered in offscreen UI checks. Refresh Flight comparison > Flights by model > M0 to see Session 1 (3 takes), Session 2 (4 takes) and Session 3 (4 takes). Raw originals are unchanged. The former Whip_123 sources were relocated by the user; their hashes match, and the relocation is documented without rewriting the historical report.
