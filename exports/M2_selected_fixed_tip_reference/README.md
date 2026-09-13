# Physical evaluation available

Five clean flights of this CSV were recorded. See
[the measured performance report](../../runs/data_review/M2-paper-20260913/REPORT.md).
The original preflight notes below are retained as history; predictions are
separate from the measured results. The CSV bytes remain unchanged.

# Selected M2 command for the next physical trial

Use `fullstate_30hz.csv` with the existing 30 Hz FullState flight program.
Launch tracked origin: (0, 0, 1.4) m. Target: (1.25, 0, 1.25) m.
Duration: 7.200 s; 217 rows. Execute the full recovery and final hold.
Save the new raw OptiTrack/controller pairs in `flight_take/`.

Selected model: updated nominal quadrotor response, retained M1 quadrotor residual,
M2 cable physical parameters and numerically verified cable residual checkpoint 100.
The regressing M2 quadrotor residual update is excluded. No further fitting ran.
The corrected spline starts from the executed M1 command and tracks the same original
M0 tip motion and timing, with the existing soft quadrotor-path penalty and slower brake.

Model selection uses existing M1_003/005 recordings; they are now development data,
not independent final evidence. The next physical flights test the corrected command.
Offline model, PVA, continuity, complete replay and UI checks passed. These checks
do not certify physical cable/propeller clearance or future tracking performance.

CSV SHA256: `c9a39713d627ceb5fe3cc555ccb2d3fdf853d3e20ae1a9c583faa661fbf8f379`
