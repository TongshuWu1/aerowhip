# Fresh normalized M1

Fitted from M0 and five cf7 flights; no retired M1 assets used. Drone nominal model, attitude response and residual; cable EI/Cb and damping plus bounded acceleration residual fitted. Source geometry/mass unchanged (157 g drone, 18 g cable assembly).

All errors use Z normalized by subtracting 0.05070575 m once. Each maneuver is excluded from its fit, but pre/post hover calibration uses all five flights. Conditional retrospective diagnostics, not independent or prospective flight evidence.

| Whip prediction RMS | M0 | M1 |
|---|---:|---:|
| Drone | 11.06 cm | 5.85 cm |
| Tip | 14.70 cm | 10.37 cm |

Tip improves in all five comparisons. Late recovery/hold drone RMS worsens from 9.56 to 15.44 cm. Whip is the priority.

Fitting completed. User stopped remaining all-five validation; no further ablations or gradient experiment were run. Existing completed full-sequence rollouts were finite. All-five candidate is for planning, and its fit evidence is in-sample. New real flights must establish policy improvement.
