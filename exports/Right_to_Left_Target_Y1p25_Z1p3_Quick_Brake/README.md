# New right-to-left strike with quick braking

Uses the M7 predictor; the original M7 horizontal command is unchanged. Starting tracked-origin hover: (0, 0, 1.5) m. Target: (0, 1.25, 1.3) m. Desired tip direction: -X.

Predicted miss 6.61 mm; tip speed 2.32 m/s; direction error 4.27 degrees. Strike at 1.3493 s, braking begins 1.3667 s, commanded deceleration lasts 0.533 s.

Maximum predicted vehicle distance from starting hover: 1.584 m horizontally. Maximum predicted distance from brake-start position during braking and the following second: 0.922 m.

Complete CSV: 333 rows at 30 Hz, 11.067 s, including braking, return, and final hold. This is not a takeoff command.

Simulation checks passed; real-flight performance is untested. Store new-task recordings under flight_take. Active flight selection has not been changed.
