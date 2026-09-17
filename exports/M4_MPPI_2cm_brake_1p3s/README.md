# M4 flight command

Use `fullstate_30hz.csv` with the existing controller and launch command.
Launch tracked origin: (0, 0, 1.4) m. Keep the established four-whip sequence with 15-second settling intervals. Execute the complete CSV, including braking, return and final hold.
CSV duration: 9.300 s.

Save the raw recordings in `flight_take/M4`. Continue the same collection procedure: no physical target and no manual cable reset between repetitions.

This command starts from the M3 command and uses the frozen M4 model to track the original M0 predicted motion at its original times. The complete simulation replay passed the saved command and model limits. Physical improvement remains unmeasured.
