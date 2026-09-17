# Four-whip recording review: M0_v2_001

The four-whip collection format is usable in this recording. All four executions match every moving command packet in the accepted M0 2 cm aiming CSV. The interval after each completed CSV is approximately 15 seconds.

The user confirms there is no physical target and no touching or resetting the cable between repetitions, for this and future supplied data. Distances below refer to the virtual planned point, not physical contact or hit success.

| Repetition | Measured tip closest distance to virtual point |
| --- | ---: |
| 1 | 9.0 cm |
| 2 | 14.3 cm |
| 3 | 13.0 cm |
| 4 | 10.8 cm |

Distances use valid measured tip segments during the first 1.5 seconds after each command onset. The virtual point is [1.25, 0, 1.25] m.

All ten cable markers pass the existing geometry and jump checks during each whipping interval. Each repetition also has a valid one-second causal cable history for initialization. Residual cable-tip motion before execution is approximately 1.1–2.5 cm RMS relative to the attachment; measured initial motion should be retained rather than assuming a motionless cable. The first hover is less steady than the later hovers. This recording does not establish statistical independence or battery safety across future flights.

The manually trimmed OptiTrack export retains timestamps from 11.69 to 97.10 seconds. Duration alone does not determine synchronization. The revised alignment uses only fresh controller-position updates: controller time = OptiTrack time + 0.746 seconds. The controller logs rows at 100 Hz but its position changes at approximately 10 Hz. The previous 0.795-second estimate was biased by repeated cached positions. Independent repetition estimates range from 0.744 to 0.748 seconds. Fresh-update XYZ agreement is 2.93 mm RMS. This is still an estimate containing unknown transport/estimator latency, not verified clock synchronization or physical actuation delay. See synchronization_review.json and synchronization_review.png.

The fourth whipping motion, braking and return are captured. The export ends approximately 0.32 seconds before the final stationary hold finishes. Keep roughly one additional second at the end when trimming future recordings; these four whips do not need to be discarded for that reason.

Keep all four repetitions grouped as one recording/battery when organizing later evaluation; do not treat them as four independent flights. No model fitting was performed and raw files were not changed. The NPZ windows are review artifacts, not registered training inputs.

The original report.json, whip NPZ windows and four_whip_review.png retain the earlier 0.795-second alignment as historical audit artifacts. Their time-dependent metrics and initialization windows are superseded and must be recomputed before fitting or formal evaluation. The UI split entries use the revised 0.746-second estimate. Native data rows remain unchanged.
