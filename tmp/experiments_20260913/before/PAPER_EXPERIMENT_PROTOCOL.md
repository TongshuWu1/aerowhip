# Current AeroWhip experiment

Preserve the current 145 g quadrotor and 17 g cable model, tracked-origin and
attachment conventions, 30 Hz PVA commands, and original M0 reference. Origin:
(0,0,1.4) m. Target: (1.25,0,1.25) m. M0 is not recollected or refitted.

Five M0 and five M1 physical recordings have been collected. The user confirmed
they are clean. Original controller/OptiTrack files, gaps, timing evidence and
flown CSVs remain unchanged. Reports are under `runs/data_review/`.

The next command is `exports/M2_selected_fixed_tip_reference/fullstate_30hz.csv`.
Store its raw pairs in the same folder's `flight_take/`. The selected M2 uses
updated nominal drone parameters, the retained M1 drone residual, and completed
M2 cable physics/residual. Keep both residuals. No further fit is scheduled.

The command correction follows the original M0 predicted physical tip motion
and timing, with a smaller quadrotor-reference penalty, initialized from the
executed M1 spline. It preserves the slower brake, return and hold. See
[the correction definition](../methods/FIXED_REFERENCE_CORRECTION.md).

Fit batches used takes 001/002/004 for adaptation and 003/005 for operational
validation. M1_003/005 were later explicitly used for development model selection;
this must be disclosed. They are not independent final tests. Missing marker
observations are masked using the existing causal initialization rules.

The previously planned final comparison remains five M0/M2 pairs: M2-M0,
M0-M2, M0-M2, M2-M0, M0-M2. Final recordings remain outside fitting, stopping
and tuning. No completed M2 flight or final comparison is claimed here.

Evaluate continuous three-dimensional target error, including error at the
original planned strike time and closest distance over the declared interval.
Also assess time-aligned tracking of the original reference. Keep retrospective
model prediction error separate from actual physical flight performance; do
not introduce a new binary success threshold or change scoring during cleanup.

The separate lab flight program executes the CSV. Offline planning is not
flight authorization. Preserve every trial and document any interruption or
tracking limitation. Windows checks do not establish Ubuntu/5080 or physical
clearance validation. The complete prior protocol is preserved in
`delete/cleanup-20260913/before_cleanup/docs/paper/PAPER_EXPERIMENT_PROTOCOL.md`.
