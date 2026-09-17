# M2 flight collection

Use `fullstate_30hz.csv` with the existing controller and launch command.
Keep the established four-whip sequence and 15-second settling intervals.
Launch tracked origin: (0, 0, 1.4) m. Execute the complete CSV, including the 1.3-second braking phase, return and final hold.
CSV duration: 8.533 seconds.

Save all raw recording files in `flight_take/M2`. These collections have no physical target and no manual cable reset between repetitions.

M2 starts from M1 and fits all 12 M1 whips with the existing preliminary training replay. Standard plateau stopping: vehicle residual 105 updates; cable residual 245 updates, selected checkpoint 240. No residual time/update cap.
Commands follow the same fixed M0 reference. The saved correction was replayed using the standard production solver, with independent complete-replay and command-envelope checks. The earlier accelerated replay discrepancy is retained in the correction provenance. Physical prediction/accuracy evaluation is deferred.
