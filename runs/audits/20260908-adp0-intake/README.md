# Real-flight adp0 intake

Five controller takes; supplied 30 Hz CSV matches saved rehearsal 20260908-203914-039721 byte-for-byte.
The scored whip is CSV time 0–1 s. The complete CSV lasts 11.2 s and includes recovery and final hold.

| Take | OptiTrack coverage relative to CSV (s) | Complete whip | Measurement-stream alignment RMS (cm) |
|---|---|---|---|
| 001 | -1.636 to 11.844 | Yes | 1.82 |
| 002 | -0.252 to 11.758 | Yes | 1.90 |
| 003 | -1.691 to 11.169 | Yes | 1.86 |
| 004 | -1.949 to 10.911 | Yes | 1.82 |
| 005 | -0.769 to 11.241 | Yes | 1.91 |

Alignment RMS is agreement between two measured position streams, **not tracking or hitting error**.

## Interpretation

- User confirmed all five flights had no object/target contact or intervention. Target assessment is geometric, not observed physical impact.
- Evaluate tip error at the saved predicted strike time separately from closest approach during the whip and its timing; do not time-shift to minimize target error.
- User identifies simulation_csv as the flown command; dynamic PVA packets match exactly in all five controller logs.
- Repeated stationary holds cannot be uniquely identified by command values alone.
- CSV onset estimated from command reception age; not a hardware actuation timestamp.
- OptiTrack alignment uses measured drone XYZ only, no desired-trajectory fit or spatial transformation.
- Alignment includes logging latency; half-window offsets differ by up to about one 100 Hz sample.
- OptiTrack wall-clock metadata has a 1969 date and is not used for synchronization.
- Only cf_7 and cable1:c1 through c10 are parsed; unlabeled markers are ignored.
- Raw logs preserved; no missing cable samples interpolated, no fit/training or old legacy phase classification.
- Take 002 was replaced during inspection: the initial trim began about 2.60 s after CSV onset; the current file begins about 0.25 s before onset and covers the whip. Hashes describe the current files.

Contact review complete: user confirms all five are free-flight trials. Fit selection remains a separate quality-review step; no fitting has been performed.
Detailed per-marker whole-take and whip validity, timing offsets and all input hashes are in intake.json.
Reproduce on the project Python environment with this directory’s inspect_batch.py.
