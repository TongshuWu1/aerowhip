# M4 collection review

12 complete whips in three recordings (4 + 4 + 4). All load individually in the UI under M4.

OptiTrack is continuous at 100 Hz during all 1.5 s review intervals. Vehicle pose and one-second initialization checks pass for every whip. 18 invalid marker samples remain masked; no whip was discarded.

Session 3 TF logger averages 92.87 Hz and has gaps. The separate command-event log retains all 187 moving packets per whip. This collection explicitly uses that log for playback and preparation; earlier protocols retain their existing command source. Synchronization remains estimated from measured position streams, not independently verified.

Raw recordings and timestamps are unchanged. No M5 fitting has started.
