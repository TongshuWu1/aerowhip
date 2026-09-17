# M5 forward-left horizontal whip — firmer brake

Same saved M5 whip, followed by a 1.0 s brake instead of 1.3 s. Only the run-specific minimum braking duration changed. Model, launch, selected release, complete whip commands, and physical command limits are unchanged. Return to the original hover and final hold remain part of the export.

Peak commanded horizontal braking acceleration increases from 4.34 to 5.24 m/s². This is a command-profile comparison, not a measured real-drone stopping time. The command is continuous in position, velocity and acceleration across the braking boundary.

Independent full drone-and-cable replay passed the existing command, attitude, height and cable-floor checks. The whip command prefix is bit-identical to the original; predicted whip cable difference is below 1e-8 m and drone difference below 1e-9 m. See braking_comparison.json for actual bounds and differences.

The virtual release point is selected from simulation; no physical contact or post-impact dynamics have been measured. The cable can continue moving after the drone brakes. Fullstate CSV includes recovery, and the active flight selection was not changed.
