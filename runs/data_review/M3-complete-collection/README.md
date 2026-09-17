# M3 collection review

12 complete whips across three battery recordings (4 + 4 + 4). All split takes load in the UI under M3; refresh Flight comparison > Flights by model.

Native OptiTrack and logger rates are approximately 100 Hz. Every 1.5 s fitting interval has continuous native OptiTrack timestamps and valid vehicle poses. All one-second cable initializations pass. Three invalid marker samples remain masked: c8 in session 1 whip 3, and the tip in session 2 whips 1 and 2. No whip was discarded.

Two logger gap flags occur during whip intervals, while native OptiTrack remains continuous. Session 1 whip 4 tracking ends at 8.195 s of the 8.9 s CSV, after the fitting interval. Alignment is estimated from the measured vehicle streams; clock synchronization remains unverified.

Raw files and native timestamps are preserved. No model fitting or performance evaluation was run. The next fit defaults to 75% all-marker plus 25% extra tip loss.
